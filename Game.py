from typing import Tuple, Dict, List

import numpy as np
from pettingzoo.atari import warlords_v3

from Agent import Agent, EXPECTED_OBSERVATION_LENGTH


class Game:
    _block_width = 8
    _block_height = 8
    _small_block_width = 4
    _min_base_pixels = 32
    _player_colours = np.array([[195, 144, 61],
                                [45, 109, 152],
                                [82, 126, 45],
                                [104, 25, 154]])
    _ball_colour = np.array([200, 72, 72])
    _ball_width = 2

    def __init__(self, agent_list: List[Agent]) -> None:
        self.agent_list = agent_list

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
                np.flip(game_area[:quadrant_height, quadrant_width:], axis=1),
                np.flip(game_area[quadrant_height:, :quadrant_width], axis=0),
                np.flip(game_area[quadrant_height:, quadrant_width:], axis=(0, 1)))

    @staticmethod
    def process_blocks(segment: np.ndarray, block_height: int, block_width: int) -> List[bool]:
        block_statuses: List[bool] = []
        segment_height, segment_width, _ = segment.shape
        assert segment_height % block_height == 0
        assert segment_width % block_width == 0
        for segment_column in np.split(segment, segment.shape[0] // block_height, axis=0):
            for block in np.split(segment_column, segment_column.shape[1] // block_width, axis=1):
                assert block.shape == (block_height, block_width, 3)
                block_statuses.append(np.all(block))
        return block_statuses

    def block_statuses(self, player_area: np.ndarray) -> np.ndarray:
        block_statuses: List[bool] = []
        # Bottom segment
        block_statuses += self.process_blocks(segment=player_area[16:40, 0:48, :],
                                              block_height=self._block_height,
                                              block_width=self._block_width)
        # Top segment
        block_statuses += self.process_blocks(segment=player_area[0:16, 32:48, :],
                                              block_height=self._block_height,
                                              block_width=self._block_width)
        # Small segment
        block_statuses += self.process_blocks(segment=player_area[0:16, 28:32, :],
                                              block_height=self._block_height,
                                              block_width=self._small_block_width)
        return np.array(block_statuses)

    @staticmethod
    def bool_blocks(area: np.ndarray) -> np.ndarray:
        # Boolean whether coordinate contains a block or not (0 or 1)
        assert area.shape[-1] == 3
        result = np.greater(np.abs(area).sum(-1), 0)
        assert result.shape == area.shape[:-1]
        return result

    def base_status(self, player_area: np.ndarray) -> np.ndarray:
        # Base is destroyed if not enough coloured pixels in area. True if base exists, false otherwise
        status = np.array([self.bool_blocks(area=player_area[0:16, 0:28, :]).sum() >= self._min_base_pixels])
        assert status.shape == (1,)
        return status

    def ball_boundary(self, game_area: np.ndarray) -> np.ndarray:
        ball_coloured_pixels = np.all(game_area == self._ball_colour, axis=-1)
        assert ball_coloured_pixels.shape == game_area.shape[:-1]
        column_sums = ball_coloured_pixels.sum(axis=0)
        ball_columns = np.argwhere(np.logical_and(column_sums != 0,
                                                  np.logical_and(column_sums != self._block_height,
                                                                 column_sums != 2 * self._block_height))).flatten()
        if ball_columns.shape == (2,):
            row_sums = ball_coloured_pixels.sum(axis=-1)
            ball_rows = np.argwhere(((row_sums - self._ball_width) % self._small_block_width) == 0).flatten()
            if ball_rows.shape == (2,):
                return np.array([ball_columns.min(), ball_rows.min(), ball_columns.max(), ball_rows.max()])
        return np.zeros(4) - 1

    def paddle_boundary(self, player_area: np.ndarray, player_index: int) -> np.ndarray:
        player_coloured_pixels = np.all(player_area == self._player_colours[player_index], axis=-1)
        assert player_coloured_pixels.shape == player_area.shape[:-1]
        player_coloured_pixels[:40, :48] = False
        xs, ys = np.where(player_coloured_pixels)
        if xs.shape == (0,) or ys.shape == (0,):
            # Paddle out of play
            result = np.zeros(4) - 1
        else:
            result = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
        assert result.shape == (4,)
        return result

    def parse_observation(self, observation: np.ndarray, agent_id: str, time: int) -> np.ndarray:
        game_area = self.get_game_area(observation)
        player_areas = self.get_player_areas(game_area)
        player_statuses = []
        for player_index, player_area in enumerate(player_areas):
            base_status = self.base_status(player_area)
            paddle_boundary = self.paddle_boundary(player_area=player_area, player_index=player_index)
            block_status = self.block_statuses(player_area)
            player_statuses.append(np.concatenate((base_status,
                                                   paddle_boundary,
                                                   block_status)))
        ball_boundary = self.ball_boundary(game_area)
        # TODO correct ball boundary transform
        if agent_id == "first_0":
            ball_boundary = ball_boundary
            ordered_player_statuses = np.concatenate(player_statuses)
        elif agent_id == "second_0":
            ball_boundary = ball_boundary
            ordered_player_statuses = np.concatenate((player_statuses[1], player_statuses[0],
                                                      player_statuses[3], player_statuses[2]))
        elif agent_id == "third_0":
            ball_boundary = ball_boundary
            ordered_player_statuses = np.concatenate((player_statuses[2], player_statuses[3],
                                                      player_statuses[0], player_statuses[1]))
        elif agent_id == "fourth_0":
            ball_boundary = ball_boundary
            ordered_player_statuses = np.concatenate((player_statuses[3], player_statuses[2],
                                                      player_statuses[1], player_statuses[0]))
        else:
            raise Exception
        parsed_observation = np.concatenate((ordered_player_statuses,
                                             ball_boundary,
                                             np.array([time])))
        assert parsed_observation.shape == (EXPECTED_OBSERVATION_LENGTH,)
        return parsed_observation

    def get_agent_dict(self, env) -> Dict[str, Agent]:
        return {agent_id: agent for agent_id, agent in zip(env.agents, self.agent_list)}

    @staticmethod
    def get_agent_times_dict(env) -> Dict[str, int]:
        return {agent_id: 0 for agent_id in env.agents}

    def get_action(self,
                   agent_dict: Dict[str, Agent],
                   agent_times_dict: Dict[str, int],
                   agent_id: str,
                   observation: np.ndarray,
                   info: dict) -> int:
        parsed_observation = self.parse_observation(observation=observation,
                                                    agent_id=agent_id,
                                                    time=agent_times_dict[agent_id])
        action = agent_dict[agent_id].action(observation=parsed_observation, info=info)
        agent_times_dict[agent_id] += 1
        # print({"Agent": agent_id,
        #        "Action": action,
        #        "Observation Shape": parsed_observation.shape,
        #        "Info": info,
        #        "Observation": parsed_observation})
        return action

    def run(self) -> None:
        env = warlords_v3.env(render_mode="human")
        env.reset(seed=42)
        agent_dict = self.get_agent_dict(env=env)
        agent_times_dict = self.get_agent_times_dict(env=env)
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                action = self.get_action(agent_dict=agent_dict,
                                         agent_times_dict=agent_times_dict,
                                         agent_id=agent,
                                         observation=observation,
                                         info=info)
            env.step(action)

        env.close()


    def run_parallel(self) -> None:
        env = warlords_v3.parallel_env(render_mode="human")
        observations, infos = env.reset()
        agent_dict = self.get_agent_dict(env=env)
        agent_times_dict = self.get_agent_times_dict(env=env)
        while env.agents:
            
            last_observations_parsed = {}
            actions = {}

            for agent in env.agents:
                last_observation_parsed = self.parse_observation(observation=observations[agent],
                                                          agent_id=agent,
                                                          time=agent_times_dict[agent]) 
                last_observations_parsed[agent] = last_observation_parsed
                action = self.get_action(agent_dict=agent_dict,
                                              agent_times_dict=agent_times_dict,
                                              agent_id=agent,
                                              observation=observations[agent],
                                              info=infos[agent])
                actions[agent] = action
            
            observations, rewards, terminations, truncations, infos = env.step(actions)

            for agent in env.agents:
                next_observation_parsed = self.parse_observation(observation=observations[agent],
                                                          agent_id=agent,
                                                          time=agent_times_dict[agent])
                last_observation_parsed = last_observations_parsed[agent]
                action = actions[agent]
                reward = rewards[agent]
                termination =  terminations[agent]
                agent_dict[agent].add_to_buffer(last_observation_parsed, action, reward, next_observation_parsed, termination)
                agent_dict[agent].train()

        env.close()
