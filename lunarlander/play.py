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

import gym
import numpy as np


class ModelLoader:
    def __init__(self, model_path):
        self._model_path = model_path
        self._model = None
        self._last_time = None
        self._last_mtime = None

    def load(self):
        current_time = time.time()
        if self._model is None or current_time > self._last_time + 2:
            current_mtime = os.path.getmtime(self._model_path)
            if current_mtime == self._last_mtime:
                return self._model
            print('{} {} ... '.format(
                "Reloading" if self._model else "Loading",
                os.path.basename(self._model_path)),
                  end='')
            try:
                self._model = pickle.load(open(self._model_path, 'rb'))
            except EOFError:
                print('file is being written to (will retry)')
            else:
                print('success')
                self._last_time = current_time
                self._last_mtime = current_mtime
        return self._model


def _make_move(agent, state):
    probs = agent.predict_proba([state])[0]
    a = np.random.choice(agent.classes_, p=probs)
    return a


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--model-file",
                        help=("The file containing the trained agent that "
                              "should be used to play the game. If not set "
                              "then the game can be controlled with arrow "
                              "keys."))

    args = parser.parse_args()

    if args.model_file:
        model_loader = ModelLoader(args.model_file)

    model = model_loader.load()
    env = gym.make(model.game)

    while True:
        model = model_loader.load()
        state = env.reset()

        while True:
            env.render()
            action = _make_move(model.agent, state)
            state, reward, done, _ = env.step(action)
            if done:
                break


if __name__ == "__main__":
    main()
