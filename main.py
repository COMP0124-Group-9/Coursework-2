import datetime
import pathlib
import time
from typing import Tuple

import numpy as np
import torch

import Network
from Agent import Agent
from Game import Game, EXPECTED_OBSERVATION_LENGTH, BLOCKS_PER_PLAYER, BALL_COORDINATE_SHAPE


def generate_reward_vector(base_status_weights: Tuple[float, float, float, float] = (24, -24 / 3, -24 / 3, -24 / 3),
                           block_status_weights: Tuple[float, float, float, float] = (1, -1 / 3, -1 / 3, -1 / 3)) \
        -> np.ndarray:
    player_rewards = []
    for player_index in range(4):
        player_rewards.append(np.concatenate(([base_status_weights[player_index]],
                                              np.zeros(4),
                                              np.ones(BLOCKS_PER_PLAYER) * block_status_weights[player_index])))
    reward_vector = np.concatenate((np.concatenate(player_rewards),
                                    np.zeros(BALL_COORDINATE_SHAPE)))
    assert reward_vector.shape == (EXPECTED_OBSERVATION_LENGTH,)
    return reward_vector


def main():
    model = Network.QNetworkFC(EXPECTED_OBSERVATION_LENGTH, len(Agent.possible_actions))
    agents = [Agent(reward_vector=generate_reward_vector(), model=model, reversed_controls=False),
              Agent(reward_vector=generate_reward_vector(), model=model, reversed_controls=True),
              Agent(reward_vector=generate_reward_vector(), model=model, reversed_controls=False),
              Agent(reward_vector=generate_reward_vector(), model=model, reversed_controls=True)]
    model_path = pathlib.Path("Models")
    if model_path.exists():
        runs = [run.stem for run in model_path.iterdir()]
        if len(runs) > 0:
            states_root = model_path / max(runs)
            print(f"Loaded from {states_root}")
            for agent_number, agent in enumerate(agents):
                agent.model.load_state_dict(torch.load(states_root / f"{agent_number}.pt"))
    try:
        game = Game(agent_list=agents)
        game_count: int = 0
        while True:
            start_time = time.time()
            print(f"Starting Game {game_count}")
            game.run_parallel()
            game_count += 1
            print(f"finished in {time.time() - start_time} seconds")
            print([agent.epsilon for agent in agents])
    except KeyboardInterrupt:
        path = model_path / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not path.parent.exists():
            path.parent.mkdir()
        path.mkdir()
        for agent_number, agent in enumerate(agents):
            torch.save(agent.model.state_dict(), path / f"{agent_number}.pt")


if __name__ == '__main__':
    main()
