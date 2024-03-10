from typing import Tuple

import numpy as np

from Agent import Agent
from Game import Game, EXPECTED_OBSERVATION_LENGTH, BLOCKS_PER_PLAYER, BALL_COORDINATE_SHAPE


def generate_reward_vector(base_status_weights: Tuple[float, float, float, float] = (1e2, -1e2, -1e2, -1e2),
                           block_status_weights: Tuple[float, float, float, float] = (5, -5, -5, -5),
                           time_weight: float = 1e-3) -> np.ndarray:
    player_rewards = []
    for player_index in range(4):
        player_rewards.append(np.concatenate(([base_status_weights[player_index]],
                                              np.zeros(4),
                                              np.ones(BLOCKS_PER_PLAYER) * block_status_weights[player_index])))
    reward_vector = np.concatenate((np.concatenate(player_rewards),
                                    np.zeros(BALL_COORDINATE_SHAPE),
                                    [time_weight]))
    assert reward_vector.shape == (EXPECTED_OBSERVATION_LENGTH,)
    return reward_vector


def main():
    agents = [Agent(reward_vector=generate_reward_vector()) for _ in range(4)]
    game = Game(agent_list=agents)
    game_count: int = 0
    while True:
        print(f"Starting Game {game_count}")
        game.run_parallel()
        game_count += 1


if __name__ == '__main__':
    main()
