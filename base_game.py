import typing
from typing import Tuple

import numpy as np
from pettingzoo.atari import warlords_v3

from Agent import Agent


class Game:
    def __init__(self, agent_list: typing.List[Agent]) -> None:
        self.agent_list = agent_list
        self.number_rounds = 0

    @staticmethod
    def get_game_area(observation: np.ndarray) -> np.ndarray:
        return observation[16:-28, :, :]

    @staticmethod
    def get_player_areas(game_area: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        play_area_shape = game_area.shape
        assert play_area_shape[0] % 2 == 0
        assert play_area_shape[1] % 2 == 0
        quadrant_height, quadrant_width = [play_area_shape[0] // 2, play_area_shape[1] // 2]
        return (game_area[:quadrant_height, :quadrant_width],
                np.flip(game_area[quadrant_height:, :quadrant_width], axis=0),
                np.flip(game_area[:quadrant_height, quadrant_width:], axis=1),
                np.flip(game_area[quadrant_height:, quadrant_width:], axis=(0, 1)))

    @staticmethod
    def block_statuses(player_area: np.ndarray) -> np.ndarray:
        # TODO can be simplified per block, not per pixel
        segment_1 = player_area[16:40, 0:28, :].reshape(-1, 3)
        segment_2 = player_area[0:40, 28:48, :].reshape(-1, 3)
        return np.sign(np.concatenate((segment_1, segment_2)).sum(axis=-1))

    @staticmethod
    def base_status(player_area: np.ndarray) -> np.ndarray:
        # Base is destroyed if all black pixels. True if base exists, false otherwise
        base = player_area[0:16, 0:28, :]
        return np.array([np.any(base)])

    @staticmethod
    def ball_boundary(player_area: np.ndarray) -> np.ndarray:
        return np.array([0, 0, 0, 0])

    @staticmethod
    def paddle_boundary(player_area: np.ndarray) -> np.ndarray:
        return np.array([0, 0, 0, 0])

    def parse_observation(self, observation: np.ndarray, agent_id: str):
        game_area = self.get_game_area(observation)
        player_areas = self.get_player_areas(observation)
        player_statuses = []
        for player_area in player_areas:
            base_status = self.base_status(player_area)
            paddle_boundary = self.paddle_boundary(player_area)
            block_status = self.block_statuses(player_area)
            player_statuses.append(np.concatenate((base_status, paddle_boundary, block_status)))
        ball_boundary = self.ball_boundary(game_area)
        # TODO correct ordering of these and transform ball boundary
        if agent_id == "first_0":
            ball_boundary = ball_boundary
            ordered_player_statuses = np.concatenate(player_statuses)
        elif agent_id == "second_0":
            ball_boundary = ball_boundary
            ordered_player_statuses = np.concatenate(player_statuses)
        elif agent_id == "third_0":
            ball_boundary = ball_boundary
            ordered_player_statuses = np.concatenate(player_statuses)
        else:
            ball_boundary = ball_boundary
            ordered_player_statuses = np.concatenate(player_statuses)
        return np.concatenate((ordered_player_statuses, ball_boundary))

    def get_agent_dict(self, env):
        return {agent_id: agent for agent_id, agent in zip(env.agents, self.agent_list)}

    def get_action(self, agent_dict, agent, observation, info):
        return agent_dict[agent].action(observation=self.parse_observation(observation, agent), info=info)

    def run(self):
        env = warlords_v3.env(render_mode="human")
        env.reset(seed=42)
        agent_dict = self.get_agent_dict(env=env)
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action = self.get_action(agent_dict=agent_dict, agent=agent, observation=observation, info=info)
            env.step(action)
        env.close()

    def run_parallel(self):
        env = warlords_v3.parallel_env(render_mode="human")
        observations, infos = env.reset()
        agent_dict = self.get_agent_dict(env=env)
        while env.agents:
            actions = {agent: self.get_action(agent_dict=agent_dict,
                                              agent=agent,
                                              observation=observations[agent],
                                              info=infos[agent])
                       for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
        env.close()
