import agent
import base_game


def main():
    agents = [agent.Agent() for _ in range(4)]
    game = base_game.Game(agent_list=agents)
    game.run()


if __name__ == '__main__':
    main()
