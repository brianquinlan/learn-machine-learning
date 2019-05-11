# Copyright 2019 Brian Quinlan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The game model for a game where a ball has to be kept in the air.

The same is similar to breakout except that there are no blocks to break at
the top of the screen i.e. the objective is simply to keep the ball in the air
using a paddle as long as possible.
"""

import abc
import enum
import random
from typing import Mapping, Optional

import numpy as np


def _normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class Direction(enum.IntEnum):
    """The possible player moves."""
    LEFT = 0
    CENTER = 1
    RIGHT = 2


class MoveMaker(abc.ABC):
    @abc.abstractmethod
    def make_move(self, state) -> Direction:
        pass

    def move_probabilities(self, state) -> Optional[Mapping[Direction, float]]:
        return None


class Game:
    NUM_STATES = 6
    NUM_ACTIONS = 3

    def __init__(self,
                 reward_time_multiplier=1.0,
                 reward_bounces_multiplier=0.0,
                 reward_height_multiplier=0.0,
                 punish_moves_multiplier=1.0):
        self._actions = []
        self._ball_pos = np.array([3 + random.random() * 19, 100.0])
        self._ball_v = np.array([-0.3 + random.random() * 0.6, 0.0])
        self.ball_radius = 1
        self._g = np.array([0.0, -0.025])
        self._paddle_pos = np.array([12.5, -9.5])
        self.paddle_radius = 10
        self._done = False
        self._score = 0
        self._reward_time_multiplier = reward_time_multiplier
        self._reward_bounces_multiplier = reward_bounces_multiplier
        self._reward_height_multiplier = reward_height_multiplier
        self._punish_moves_multiplier = punish_moves_multiplier

    @property
    def done(self):
        return self._done

    @property
    def ball_x(self):
        return self._ball_pos[0]

    @property
    def ball_y(self):
        return self._ball_pos[1]

    @property
    def paddle_x(self):
        return self._paddle_pos[0]

    @property
    def paddle_y(self):
        return self._paddle_pos[1]

    @property
    def state(self):
        return (
            self._ball_pos[0],
            self._ball_pos[1],
            self._ball_v[0],
            self._ball_v[1],
            self._paddle_pos[0],
            self._paddle_pos[1],
        )

    @property
    def score(self):
        return self._score

    def _update(self):
        assert not self._done

        self._score += self._reward_time_multiplier

        old_y = self.ball_y
        self._ball_v += self._g
        self._ball_pos += self._ball_v

        distance = np.linalg.norm(
            [self.paddle_x - self.ball_x, self.paddle_y - self.ball_y])
        if distance < (self.paddle_radius + self.ball_radius):
            self._score += self._reward_bounces_multiplier
            n = _normalize(
                [self.paddle_x - self.ball_x, self.paddle_y - self.ball_y])
            self._ball_v = self._ball_v - 2 * (np.dot(self._ball_v, n)) * n
            self._ball_pos += self._ball_v

        if self.ball_y < self.ball_radius:
            self._done = True
        elif self.ball_x < self.ball_radius:
            self._score += self._reward_bounces_multiplier
            self._ball_v *= [-1, 1]
            self._ball_pos += [self._ball_v[0], 0]
        elif self.ball_x > (25 - self.ball_radius):
            self._score += self._reward_bounces_multiplier
            self._ball_v *= [-1, 1]
            self._ball_pos += [self._ball_v[0], 0]

        if self.ball_y > old_y:
            self._score += (self.ball_y -
                            old_y) * self._reward_height_multiplier

        return self.state

    def move_left(self):
        self._score -= self._punish_moves_multiplier
        self._paddle_pos += [-2, 0]
        if self._paddle_pos[0] < 0:
            self._paddle_pos[0] = 0
        return self._update()

    def move_right(self):
        self._score -= self._punish_moves_multiplier
        self._paddle_pos += [2, 0]
        if self._paddle_pos[0] > 25:
            self._paddle_pos[0] = 25
        return self._update()

    def stay(self):
        return self._update()

    def move(self, direction: Direction):
        if direction == Direction.LEFT:
            self.move_left()
        elif direction == Direction.CENTER:
            self.stay()
        elif direction == Direction.RIGHT:
            self.move_right()
        else:
            assert "unexpected direction: %r" % (direction)
