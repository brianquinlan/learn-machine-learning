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
"""Play the "Well Bouncer Game" using either an agent or interactively."""

import argparse
import os.path
import pickle
import time
from typing import Mapping, Optional

import numpy as np

import pygame_well_bouncer_player
import well_bouncer_game


class AgentMoveMaker(well_bouncer_game.MoveMaker):
    def __init__(self, model_path):
        self._model_path = model_path
        self._agent = None
        self._last_time = None
        self._last_mtime = None

    def _load_agent(self):
        current_time = time.time()
        if self._agent is None or current_time > self._last_time + 2:
            current_mtime = os.path.getmtime(self._model_path)
            if current_mtime == self._last_mtime:
                return
            print('{} {} ... '.format(
                "Reloading" if self._agent else "Loading",
                os.path.basename(self._model_path)),
                  end='')
            try:
                self._agent = pickle.load(open(self._model_path, 'rb'))
            except EOFError:
                print('file is being written to (will retry)')
            else:
                print('success')
                self._last_time = current_time
                self._last_mtime = current_mtime

    def make_move(self, state) -> well_bouncer_game.Direction:
        self._load_agent()
        probs = self._agent.predict_proba([state])[0]
        action = np.random.choice(self._agent.classes_, p=probs)
        return action

    def move_probabilities(
            self,
            state) -> Optional[Mapping[well_bouncer_game.Direction, float]]:
        self._load_agent()
        return {
            move: prob
            for (move, prob) in zip(self._agent.classes_,
                                    self._agent.predict_proba([state])[0])
        }


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--model-file",
                        help=("The file containing the trained agent that "
                              "should be used to play the game. If not set "
                              "then the game can be controlled with arrow "
                              "keys."))

    args = parser.parse_args()

    if args.model_file:
        move_maker = AgentMoveMaker(args.model_file)
        title = os.path.basename(args.model_file)
    else:
        move_maker = pygame_well_bouncer_player.InteractiveMoveMaker()
        title = 'You'

    while True:
        game = well_bouncer_game.Game()
        pygame_well_bouncer_player.play_game(title, game, move_maker)


if __name__ == "__main__":
    main()
