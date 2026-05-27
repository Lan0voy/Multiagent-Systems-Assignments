"""
Практическое задание по теме Ray / RLlib MultiAgentEnv.

Программа реализует простую мультиагентную среду "Камень, бумага, ножницы".
Два агента действуют одновременно. После каждого раунда среда возвращает
словарь наблюдений, словарь наград и признаки завершения эпизода.

Основной режим запуска использует Ray Core: несколько эпизодов моделируются
параллельно как удалённые задачи Ray.

Примеры запуска:
    python ray_multiagent_rps.py --mode ray --episodes 100 --rounds 10
    python ray_multiagent_rps.py --mode local --episodes 10 --rounds 5
"""

import argparse
import random
from collections import Counter

# -----------------------------------------------------------------------------
# Необязательные импорты.
# Если ray[rllib] установлен, среда наследуется от настоящего MultiAgentEnv.
# Если библиотека не установлена, локальный режим всё равно сможет показать
# логику среды, но Ray-режим потребует установки пакета ray[rllib].
# -----------------------------------------------------------------------------
try:
    import gymnasium as gym
except ImportError:
    gym = None

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:
    class MultiAgentEnv:
        """Минимальная заглушка для локального запуска без установленного Ray."""
        pass


class SimpleDiscrete:
    """Простая замена gymnasium.spaces.Discrete для локального режима."""

    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randrange(self.n)


class RockPaperScissorsEnv(MultiAgentEnv):
    """
    Мультиагентная среда для игры "Камень, бумага, ножницы".

    В среде есть два агента:
        agent_1
        agent_2

    Каждый агент на каждом шаге выбирает одно действие:
        0 - камень
        1 - бумага
        2 - ножницы

    Оба агента действуют одновременно. Награда назначается после сравнения ходов.
    """

    ROCK = 0
    PAPER = 1
    SCISSORS = 2

    ACTION_NAMES = {
        ROCK: "камень",
        PAPER: "бумага",
        SCISSORS: "ножницы",
    }

    # Матрица выигрышей: ключ - пара действий, значение - награды двух агентов.
    # Камень бьёт ножницы, ножницы бьют бумагу, бумага бьёт камень.
    WIN_MATRIX = {
        (ROCK, ROCK): (0, 0),
        (ROCK, PAPER): (-1, 1),
        (ROCK, SCISSORS): (1, -1),
        (PAPER, ROCK): (1, -1),
        (PAPER, PAPER): (0, 0),
        (PAPER, SCISSORS): (-1, 1),
        (SCISSORS, ROCK): (-1, 1),
        (SCISSORS, PAPER): (1, -1),
        (SCISSORS, SCISSORS): (0, 0),
    }

    def __init__(self, config=None):
        """Создаёт среду и задаёт множество агентов и пространств действий."""
        super().__init__()

        if config is None:
            config = {}

        # Максимальное количество раундов в одном эпизоде.
        self.max_rounds = int(config.get("max_rounds", 10))

        # В RLlib список possible_agents хранит всех агентов,
        # которые потенциально могут появиться в эпизоде.
        self.possible_agents = ["agent_1", "agent_2"]

        # В этой задаче агенты не исчезают до конца эпизода,
        # поэтому active agents равны possible_agents.
        self.agents = list(self.possible_agents)

        # Если gymnasium доступен, используем его пространства.
        # Иначе используем простую локальную замену.
        discrete_space = gym.spaces.Discrete(3) if gym is not None else SimpleDiscrete(3)

        # Наблюдение агента - последний ход противника.
        # Действие агента - один из трёх вариантов: камень, бумага, ножницы.
        self.observation_spaces = {
            "agent_1": discrete_space,
            "agent_2": discrete_space,
        }
        self.action_spaces = {
            "agent_1": discrete_space,
            "agent_2": discrete_space,
        }

        # Счётчик раундов внутри эпизода.
        self.round_number = 0

        # Последние действия агентов. До первого хода считаем, что оба показали камень.
        self.last_actions = {
            "agent_1": self.ROCK,
            "agent_2": self.ROCK,
        }

    def reset(self, *, seed=None, options=None):
        """
        Сбрасывает среду перед новым эпизодом.

        Возвращает:
            observations - словарь наблюдений для агентов, которые должны действовать;
            infos        - служебная информация.
        """
        if seed is not None:
            random.seed(seed)

        self.round_number = 0
        self.agents = list(self.possible_agents)
        self.last_actions = {
            "agent_1": self.ROCK,
            "agent_2": self.ROCK,
        }

        # Оба агента должны сделать ход уже на первом шаге,
        # поэтому наблюдения выдаются сразу двум агентам.
        observations = {
            "agent_1": self.last_actions["agent_2"],
            "agent_2": self.last_actions["agent_1"],
        }

        infos = {
            "agent_1": {},
            "agent_2": {},
        }

        return observations, infos

    def step(self, action_dict):
        """
        Выполняет один шаг среды.

        action_dict содержит действия агентов:
            {
                "agent_1": действие первого агента,
                "agent_2": действие второго агента
            }
        """
        self.round_number += 1

        # Получаем действия обоих агентов.
        action_1 = int(action_dict["agent_1"])
        action_2 = int(action_dict["agent_2"])

        # Сохраняем последние действия.
        self.last_actions = {
            "agent_1": action_1,
            "agent_2": action_2,
        }

        # Считаем награды по матрице выигрышей.
        reward_1, reward_2 = self.WIN_MATRIX[(action_1, action_2)]

        rewards = {
            "agent_1": reward_1,
            "agent_2": reward_2,
        }

        # Наблюдение каждого агента - действие противника в прошлом раунде.
        observations = {
            "agent_1": action_2,
            "agent_2": action_1,
        }

        # Эпизод завершается после max_rounds раундов.
        done = self.round_number >= self.max_rounds

        # В RLlib ключ "__all__" означает завершение эпизода для всех агентов.
        terminateds = {
            "agent_1": done,
            "agent_2": done,
            "__all__": done,
        }

        # Принудительного обрыва по лимиту времени здесь нет,
        # поэтому truncateds везде False.
        truncateds = {
            "agent_1": False,
            "agent_2": False,
            "__all__": False,
        }

        # В infos добавляем текстовое описание действий для удобства анализа.
        infos = {
            "agent_1": {
                "own_action": self.ACTION_NAMES[action_1],
                "opponent_action": self.ACTION_NAMES[action_2],
                "round": self.round_number,
            },
            "agent_2": {
                "own_action": self.ACTION_NAMES[action_2],
                "opponent_action": self.ACTION_NAMES[action_1],
                "round": self.round_number,
            },
        }

        return observations, rewards, terminateds, truncateds, infos


def choose_random_actions(env, observations):
    """Выбирает случайные действия для всех агентов, которым выданы наблюдения."""
    actions = {}

    for agent_id in observations:
        # Если action_space поддерживает sample(), используем его.
        actions[agent_id] = env.action_spaces[agent_id].sample()

    return actions


def run_episode(episode_id, max_rounds, seed):
    """Запускает один эпизод игры со случайными агентами."""
    env = RockPaperScissorsEnv({"max_rounds": max_rounds})

    observations, infos = env.reset(seed=seed + episode_id)

    total_rewards = {
        "agent_1": 0,
        "agent_2": 0,
    }

    transcript = []

    while True:
        # Оба агента получают наблюдения, значит оба должны сделать действие.
        actions = choose_random_actions(env, observations)

        observations, rewards, terminateds, truncateds, infos = env.step(actions)

        # Накопление суммарных наград.
        total_rewards["agent_1"] += rewards["agent_1"]
        total_rewards["agent_2"] += rewards["agent_2"]

        transcript.append({
            "round": infos["agent_1"]["round"],
            "agent_1_action": infos["agent_1"]["own_action"],
            "agent_2_action": infos["agent_2"]["own_action"],
            "agent_1_reward": rewards["agent_1"],
            "agent_2_reward": rewards["agent_2"],
        })

        # Проверяем завершение эпизода.
        if terminateds.get("__all__", False) or truncateds.get("__all__", False):
            break

    if total_rewards["agent_1"] > total_rewards["agent_2"]:
        winner = "agent_1"
    elif total_rewards["agent_2"] > total_rewards["agent_1"]:
        winner = "agent_2"
    else:
        winner = "draw"

    return {
        "episode": episode_id,
        "total_rewards": total_rewards,
        "winner": winner,
        "transcript": transcript,
    }


def print_episode_details(result):
    """Печатает подробный ход первого эпизода."""
    print("\nПодробный ход первого эпизода:")
    print("Раунд | agent_1 | agent_2 | награда agent_1 | награда agent_2")
    print("-" * 68)

    for row in result["transcript"]:
        print(
            f"{row['round']:>5} | "
            f"{row['agent_1_action']:<8} | "
            f"{row['agent_2_action']:<8} | "
            f"{row['agent_1_reward']:>15} | "
            f"{row['agent_2_reward']:>15}"
        )


def print_summary(results, mode, episodes, rounds):
    """Печатает итоговую статистику по всем эпизодам."""
    winners = Counter(result["winner"] for result in results)

    total_agent_1 = sum(result["total_rewards"]["agent_1"] for result in results)
    total_agent_2 = sum(result["total_rewards"]["agent_2"] for result in results)

    print("\nАнализ мультиагентной среды Ray/RLlib")
    print("Среда: Камень, бумага, ножницы")
    print(f"Режим запуска: {mode}")
    print(f"Количество эпизодов: {episodes}")
    print(f"Раундов в одном эпизоде: {rounds}")
    print("Политика агентов: случайный выбор действия")

    print("\nРезультаты:")
    print(f"- побед agent_1: {winners.get('agent_1', 0)}")
    print(f"- побед agent_2: {winners.get('agent_2', 0)}")
    print(f"- ничьих: {winners.get('draw', 0)}")

    print("\nСуммарные награды:")
    print(f"- agent_1: {total_agent_1}")
    print(f"- agent_2: {total_agent_2}")


def main():
    """Точка входа в программу."""
    parser = argparse.ArgumentParser(
        description="Демонстрация Ray/RLlib MultiAgentEnv на игре Камень, бумага, ножницы."
    )

    parser.add_argument(
        "--mode",
        choices=["ray", "local"],
        default="ray",
        help="ray - параллельный запуск через Ray; local - последовательный запуск без Ray",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Количество эпизодов моделирования",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Количество раундов в одном эпизоде",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Начальное значение генератора случайных чисел",
    )

    args = parser.parse_args()

    if args.mode == "ray":
        try:
            import ray
        except ImportError:
            print("Ошибка: Ray не установлен.")
            print("Установи пакет командой:")
            print('    pip install -U "ray[rllib]"')
            print("Для проверки логики без Ray можно запустить:")
            print("    python ray_multiagent_rps.py --mode local")
            return

        # Инициализируем Ray. В локальном режиме Ray создаёт воркеры на текущем компьютере.
        ray.init(ignore_reinit_error=True)

        # Превращаем обычную функцию run_episode в удалённую Ray-задачу.
        remote_run_episode = ray.remote(run_episode)

        # Запускаем эпизоды параллельно.
        futures = [
            remote_run_episode.remote(i, args.rounds, args.seed)
            for i in range(args.episodes)
        ]

        # Получаем результаты всех удалённых задач.
        results = ray.get(futures)

        # Завершаем работу Ray.
        ray.shutdown()
    else:
        # Последовательный режим нужен для быстрой проверки логики среды.
        results = [
            run_episode(i, args.rounds, args.seed)
            for i in range(args.episodes)
        ]

    print_summary(results, args.mode, args.episodes, args.rounds)

    if results:
        print_episode_details(results[0])


if __name__ == "__main__":
    main()
