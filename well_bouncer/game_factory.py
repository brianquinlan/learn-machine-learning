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
"""Create new well_bouncer_game instances based on configuration options."""

import well_bouncer_game


class GameFactory:
    def __init__(self,
                 reward_time_multiplier=1.0,
                 reward_bounces_multiplier=0.0,
                 reward_height_multiplier=0.0,
                 punish_moves_multiplier=1.0):
        self._reward_time_multiplier = reward_time_multiplier
        self._reward_bounces_multiplier = reward_bounces_multiplier
        self._reward_height_multiplier = reward_height_multiplier
        self._punish_moves_multiplier = punish_moves_multiplier

    def new_game(self):
        return well_bouncer_game.Game(
            reward_time_multiplier=self._reward_time_multiplier,
            reward_bounces_multiplier=self._reward_bounces_multiplier,
            reward_height_multiplier=self._reward_height_multiplier,
            punish_moves_multiplier=self._punish_moves_multiplier)
