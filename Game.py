from typing import Tuple, Dict, List

import numpy as np
import supersuit
from pettingzoo.atari import warlords_v3

from Agent import Agent, EXPECTED_OBSERVATION_LENGTH

BLOCKS_PER_PLAYER = 24
BALL_COORDINATE_SHAPE = 4


class Game:
    _block_width = 8
    _block_height = 8
    _small_block_width = 4
    _min_base_pixels = 32
    _player_colours = np.array([[195, 144, 61],
                                [45, 109, 152],
                                [82, 126, 45],
                                [104, 25, 154]])
    _not_ball_colours = np.concatenate((_player_colours,
                                        [[0, 0, 0],
                                         [170, 170, 170]]))
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
        for segment_row in np.split(segment, segment.shape[0] // block_height, axis=0):
            for block in np.split(segment_row, segment_row.shape[1] // block_width, axis=1):
                assert block.shape == (block_height, block_width, 3)
                block_statuses.append(np.any(block))
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
        assert len(block_statuses) == BLOCKS_PER_PLAYER
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
        ball_coloured_pixels = np.logical_not(np.all(np.isin(game_area, self._not_ball_colours), axis=-1))
        assert ball_coloured_pixels.shape == game_area.shape[:-1]
        column_sums = ball_coloured_pixels.sum(axis=0)
        ball_columns = np.argwhere(np.logical_and(column_sums != 0, column_sums % self._block_height != 0)).flatten()
        boundary = None
        if ball_columns.shape != (0,):
            row_sums = ball_coloured_pixels.sum(axis=-1)
            ball_rows = np.argwhere((row_sums % self._small_block_width) != 0).flatten()
            if ball_rows.shape != (0,):
                boundary = np.array([ball_columns.min(), ball_rows.min(), ball_columns.max() + 1, ball_rows.max() + 1])
        if boundary is None:
            boundary = np.zeros(BALL_COORDINATE_SHAPE) - 1
        assert boundary.shape == (BALL_COORDINATE_SHAPE,)
        return boundary

    def paddle_boundary(self, player_area: np.ndarray, player_index: int) -> np.ndarray:
        player_coloured_pixels = np.all(player_area == self._player_colours[player_index], axis=-1)
        assert player_coloured_pixels.shape == player_area.shape[:-1]
        player_coloured_pixels[:40, :48] = False
        xs, ys = np.where(player_coloured_pixels)
        if xs.shape == (0,) or ys.shape == (0,):
            # Paddle out of play
            result = np.zeros(4) - 1
        else:
            result = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        assert result.shape == (4,)
        return result

    def parse_observation(self,
                          observation: np.ndarray,
                          agent_id: str,
                          last_ball_position: np.ndarray,
                          last_paddle_positions: List[np.ndarray]) -> np.ndarray:
        game_area = self.get_game_area(observation)
        player_areas = self.get_player_areas(game_area)
        player_statuses = []
        for player_index, player_area in enumerate(player_areas):
            base_status = self.base_status(player_area)
            paddle_boundary = self.paddle_boundary(player_area=player_area, player_index=player_index)
            block_status = self.block_statuses(player_area)
            player_statuses.append(np.concatenate((base_status,
                                                   paddle_boundary,
                                                   paddle_boundary - last_paddle_positions[player_index],
                                                   block_status)))
        ball_boundary = self.ball_boundary(game_area)
        # TODO correct ball boundary transform
        if agent_id == "first_0":
            ball_boundary = ball_boundary
            ordered_player_statuses = np.concatenate(player_statuses)
        elif agent_id == "second_0":
            ball_boundary = [160 - ball_boundary[2], ball_boundary[1], 160 - ball_boundary[0], ball_boundary[3]]
            ordered_player_statuses = np.concatenate((player_statuses[1], player_statuses[0],
                                                      player_statuses[3], player_statuses[2]))
        elif agent_id == "third_0":
            ball_boundary = [ball_boundary[0], 166 - ball_boundary[3], ball_boundary[2], 166 - ball_boundary[1]]
            ordered_player_statuses = np.concatenate((player_statuses[2], player_statuses[3],
                                                      player_statuses[0], player_statuses[1]))
        elif agent_id == "fourth_0":
            ball_boundary = [160 - ball_boundary[2], 166 - ball_boundary[3], 160 - ball_boundary[0],
                             166 - ball_boundary[1]]
            ordered_player_statuses = np.concatenate((player_statuses[3], player_statuses[2],
                                                      player_statuses[1], player_statuses[0]))
        else:
            raise Exception
        parsed_observation = np.concatenate((ordered_player_statuses,
                                             ball_boundary,
                                             ball_boundary - last_ball_position))
        assert parsed_observation.shape == (EXPECTED_OBSERVATION_LENGTH,)
        return parsed_observation

    def get_agent_dict(self, agent_ids: List[str]) -> Dict[str, Agent]:
        return {agent_id: agent for agent_id, agent in zip(agent_ids, self.agent_list)}

    @staticmethod
    def get_action(agent_dict: Dict[str, Agent], agent_id: str, parsed_observation: np.ndarray) -> int:
        action = agent_dict[agent_id].action(observation=parsed_observation)
        assert action in Agent.possible_actions
        return action

    def run(self) -> None:
        env = warlords_v3.parallel_env(render_mode="human", full_action_space=True)
        env = supersuit.frame_skip_v0(env, 4)
        observations, _ = env.reset()
        agent_ids = env.agents
        assert len(agent_ids) == len(self.agent_list)
        agent_dict = self.get_agent_dict(agent_ids=agent_ids)
        last_ball_positions = {agent_id: np.array([-1, -1, -1, -1]) for agent_id in agent_ids}
        last_paddle_positions = {agent_id: [np.array([-1, -1, -1, -1]) for _ in range(len(agent_ids))]
                                 for agent_id in agent_ids}
        last_observations_parsed = {agent: self.parse_observation(observation=observations[agent],
                                                                  agent_id=agent,
                                                                  last_ball_position=last_ball_positions[agent],
                                                                  last_paddle_positions=last_paddle_positions[agent])
                                    for agent in agent_ids}
        final_observations = {}

        while env.agents:
            actions = {agent: self.get_action(agent_dict=agent_dict,
                                              agent_id=agent,
                                              parsed_observation=last_observations_parsed[agent])
                       for agent in agent_ids}
            for agent in agent_ids:
                last_paddle_positions[agent][0] = last_observations_parsed[agent][1:5]
                last_paddle_positions[agent][1] = last_observations_parsed[agent][34:38]
                last_paddle_positions[agent][2] = last_observations_parsed[agent][67:71]
                last_paddle_positions[agent][3] = last_observations_parsed[agent][100:104]
                last_ball_positions[agent] = last_observations_parsed[agent][-8:-4]

            observations, _, terminations, _, _ = env.step(actions)

            for agent in agent_ids:
                if agent in observations.keys():
                    next_observation_parsed = self.parse_observation(observation=observations[agent],
                                                                     agent_id=agent,
                                                                     last_ball_position=last_ball_positions[agent],
                                                                     last_paddle_positions=last_paddle_positions[agent])
                    termination = terminations[agent]
                    reward = agent_dict[agent].reward(observation=next_observation_parsed)
                    if termination:
                        final_observations[agent] = observations[agent]
                else:
                    next_observation_parsed = last_observations_parsed[agent]
                    next_observation_parsed[0] = 0
                    termination = True
                    reward = -1e20
                    observations[agent] = final_observations[agent]
                agent_dict[agent].add_to_buffer(last_observations_parsed[agent],
                                                actions[agent],
                                                reward,
                                                next_observation_parsed,
                                                termination)
                agent_dict[agent].train()
                last_observations_parsed[agent] = next_observation_parsed
        env.close()
