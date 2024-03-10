from Agent import Agent
from Game import Game


def main():
    agents = [Agent() for _ in range(4)]
    game = Game(agent_list=agents)
    game_count: int = 0
    while True:
        print(f"Starting Game {game_count}")
        game.run_parallel()
        game_count += 1


if __name__ == '__main__':
    main()
