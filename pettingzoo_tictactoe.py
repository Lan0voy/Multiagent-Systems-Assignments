"""
Задание 2. PettingZoo: запуск и анализ мультиагентной среды TicTacToe.

Что делает программа:
1. Создаёт среду TicTacToe из библиотеки PettingZoo.
2. Запускает несколько партий между двумя случайными агентами.
3. В каждой партии агенты выбирают только допустимые ходы.
4. Собирает статистику: победы player_1, победы player_2, ничьи, суммарные награды.
5. При необходимости показывает одну партию визуально через render_mode="human".

python -m pip install "pettingzoo[classic]" - установить библиотеку

Запуск без визуализации:
    python pettingzoo_tictactoe.py --episodes 100

Запуск одной визуальной партии:
    python pettingzoo_tictactoe.py --episodes 1 --render
"""

# argparse нужен для чтения параметров из командной строки:
# --episodes, --render, --seed.
import argparse

# random нужен для случайного выбора хода агентом.
import random

# Counter нужен для удобного подсчёта побед, поражений и ничьих.
from collections import Counter

# Импортируем классическую среду TicTacToe из PettingZoo.
# Это готовая мультиагентная среда крестиков-ноликов.
from pettingzoo.classic import tictactoe_v3


def choose_random_valid_action(observation, rng):
    """
    Выбирает случайное допустимое действие для текущего агента.

    В PettingZoo TicTacToe наблюдение observation является словарём.
    В нём есть поле "action_mask".

    action_mask — это список из 9 элементов:
    - 1 означает, что клетка свободна и туда можно ходить;
    - 0 означает, что клетка занята и ход туда запрещён.

    Позиции соответствуют клеткам поля 3x3:

        0 | 1 | 2
        ---------
        3 | 4 | 5
        ---------
        6 | 7 | 8
    """

    # Получаем маску допустимых действий.
    action_mask = observation["action_mask"]

    # Создаём список всех действий, которые разрешены.
    valid_actions = []

    # enumerate даёт одновременно номер действия и значение в маске.
    for action_id, is_allowed in enumerate(action_mask):
        # Если is_allowed == 1, значит действие разрешено.
        if is_allowed == 1:
            valid_actions.append(action_id)

    # Случайно выбираем одно действие из списка допустимых.
    return rng.choice(valid_actions)


def run_one_episode(episode_id, seed=42, render=False):
    """
    Запускает одну партию TicTacToe.

    episode_id:
        номер эпизода. Используется для изменения seed,
        чтобы партии не были одинаковыми.

    seed:
        базовое зерно генератора случайных чисел.

    render:
        если True, PettingZoo откроет визуальное окно игры.
        Для статистики render лучше держать False.
    """

    # Создаём отдельный генератор случайных чисел.
    # Так результаты можно воспроизводить при одинаковом seed.
    rng = random.Random(seed + episode_id)

    # Создаём среду.
    # render_mode="human" открывает графическое окно.
    # render_mode=None запускает игру без окна, быстрее и стабильнее.
    env = tictactoe_v3.env(render_mode="human" if render else None)

    # Сбрасываем среду перед началом новой партии.
    # В новых версиях PettingZoo reset может принимать seed.
    # В старых версиях может не принимать, поэтому используем try/except.
    try:
        env.reset(seed=seed + episode_id)
    except TypeError:
        env.reset()

    # Словарь для суммарных наград игроков за всю партию.
    total_rewards = {
        "player_1": 0,
        "player_2": 0
    }

    # Счётчик ходов.
    moves_count = 0

    # agent_iter() возвращает агентов по очереди.
    # В TicTacToe сначала ходит один агент, потом другой, и так далее.
    for agent in env.agent_iter():
        # env.last() возвращает данные для текущего активного агента:
        # observation  - наблюдение текущего агента;
        # reward       - награда текущего агента за предыдущий переход;
        # termination  - естественное завершение игры;
        # truncation   - принудительное завершение, например по лимиту;
        # info         - дополнительная служебная информация.
        observation, reward, termination, truncation, info = env.last()

        # Добавляем награду текущему агенту.
        total_rewards[agent] += reward

        # Если игра уже завершилась, действие должно быть None.
        # Это требование API PettingZoo.
        if termination or truncation:
            action = None
        else:
            # Иначе выбираем случайный допустимый ход.
            action = choose_random_valid_action(observation, rng)

            # Увеличиваем счётчик реальных ходов.
            moves_count += 1

        # Передаём действие в среду.
        # После этого среда переключится на следующего агента.
        env.step(action)

    # Закрываем среду после завершения партии.
    # Это особенно важно при render_mode="human", чтобы закрыть окно.
    env.close()

    # Определяем победителя по суммарной награде.
    # В TicTacToe победитель получает большую итоговую награду.
    if total_rewards["player_1"] > total_rewards["player_2"]:
        winner = "player_1"
    elif total_rewards["player_2"] > total_rewards["player_1"]:
        winner = "player_2"
    else:
        winner = "draw"

    # Возвращаем результат партии в виде словаря.
    return {
        "episode_id": episode_id,
        "winner": winner,
        "player_1_reward": total_rewards["player_1"],
        "player_2_reward": total_rewards["player_2"],
        "moves_count": moves_count
    }


def print_results(results):
    """
    Выводит общую статистику по всем запущенным партиям.
    """

    # Считаем количество побед каждого результата.
    winner_counter = Counter(result["winner"] for result in results)

    # Считаем суммарные награды игроков.
    player_1_total = sum(result["player_1_reward"] for result in results)
    player_2_total = sum(result["player_2_reward"] for result in results)

    # Считаем среднее число ходов.
    average_moves = sum(result["moves_count"] for result in results) / len(results)

    print("\n=== Результаты эксперимента PettingZoo TicTacToe ===\n")

    print("Количество партий:", len(results))
    print("Политика агентов: случайный выбор допустимого хода")
    print()

    print("Победы и ничьи:")
    print("player_1:", winner_counter.get("player_1", 0))
    print("player_2:", winner_counter.get("player_2", 0))
    print("ничья:", winner_counter.get("draw", 0))
    print()

    print("Суммарные награды:")
    print("player_1:", player_1_total)
    print("player_2:", player_2_total)
    print()

    print("Среднее количество ходов за партию:", round(average_moves, 2))


def main():
    """
    Главная функция программы.

    Она:
    1. Читает параметры командной строки.
    2. Запускает нужное количество партий.
    3. Выводит статистику.
    """

    # Создаём объект парсера аргументов командной строки.
    parser = argparse.ArgumentParser(
        description="PettingZoo TicTacToe: случайные агенты"
    )

    # --episodes задаёт количество партий.
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Количество партий для запуска"
    )

    # --seed задаёт базовое зерно случайности.
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Зерно генератора случайных чисел"
    )

    # --render включает визуальное отображение.
    parser.add_argument(
        "--render",
        action="store_true",
        help="Показать визуальное окно игры"
    )

    # Читаем аргументы.
    args = parser.parse_args()

    # Если включён render и пользователь просит много партий,
    # предупреждаем, что лучше показывать только одну.
    # Иначе окна будут открываться/закрываться много раз. Это не смерть, но близко.
    if args.render and args.episodes > 1:
        print("Включён --render. Для визуального режима лучше запускать 1 партию.")
        print("Программа всё равно продолжит выполнение, но это может быть неудобно.\n")

    # Список для хранения результатов всех партий.
    results = []

    # Запускаем партии.
    for episode_id in range(args.episodes):
        result = run_one_episode(
            episode_id=episode_id,
            seed=args.seed,
            render=args.render
        )

        results.append(result)

    # Выводим итоговую статистику.
    print_results(results)


# Эта проверка означает:
# запускать main() только если файл запущен напрямую,
# а не импортирован как модуль.
if __name__ == "__main__":
    main()
