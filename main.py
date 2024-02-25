from Agent import Agent
from base_game import Game


def main():
    agents = [Agent() for _ in range(4)]
    game = Game(agent_list=agents)
    game.run_parallel()


if __name__ == '__main__':
    main()
