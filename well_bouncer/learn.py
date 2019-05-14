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
"""Learn how to play "Well Bouncer Game" using machine learning."""

import argparse
import os.path
import pickle
import signal
import warnings

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import sklearn.exceptions
import tqdm

import game_factory
import model
import well_bouncer_game


def _make_move(agent, g):
    probs = agent.predict_proba([g.state])[0]
    a = np.random.choice(agent.classes_, p=probs)
    g.move(a)
    return a


def _generate_session(agent, game_factory, t_max=100000):
    states, actions = [], []
    g = game_factory.new_game()

    for t in range(t_max):
        states.append(g.state)
        actions.append(_make_move(agent, g))

        if g.done:
            break
    return states, actions, g.score


def _select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    reward_threshold = np.percentile(rewards_batch, percentile)
    elite_reward_indices = [
        i for i in range(len(rewards_batch))
        if rewards_batch[i] > reward_threshold
    ]

    elite_states = []
    elite_actions = []
    for i in elite_reward_indices:
        elite_states.extend(states_batch[i])
        elite_actions.extend(actions_batch[i])

    return elite_states, elite_actions


def _simple_train_once(agent, num_games):
    for i in range(num_games):
        g = well_bouncer_game.Game()
        states = []
        actions = []
        for t in range(10000):
            states.append(g.state)
            if g.ball_x < g.paddle_x:
                actions.append(0)
                g.move_left()
            elif g.ball_x > g.paddle_x:
                actions.append(2)
                g.move_right()
            else:
                actions.append(1)
                g.stay()
            if g.done:
                break

        agent.fit(states, actions)


def _self_train_once(agent, game_factory, num_games, elite_percentile,
                     stop_checker):
    states = []
    actions = []
    scores = []
    for _ in tqdm.tqdm(range(num_games),
                       leave=False,
                       ncols=31,
                       bar_format="{bar}"):
        state, a, score = _generate_session(agent, game_factory)
        if stop_checker():
            return
        states.append(state)
        actions.append(a)
        scores.append(score)

    elite_states, elite_actions = _select_elites(states, actions, scores,
                                                 elite_percentile)

    print("{:10.1f} {:10.1f} {:10}".format(
        np.mean(scores),
        np.percentile(scores, elite_percentile),
        len(elite_states),
    ))

    if elite_states and elite_actions:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=sklearn.exceptions.ConvergenceWarning)
            agent.fit(elite_states, elite_actions)


def main():
    stop = False

    def quit_handler(signum, frame):
        nonlocal stop
        stop = True
        print("Quitting...")

    signal.signal(signal.SIGINT, quit_handler)

    parser = argparse.ArgumentParser(
        description='Learn how to play "Well Bouncer".')
    parser.add_argument("--model-file",
                        required=True,
                        help=("The file to use when loading and saving the "
                              "agent trained using machine learning. If the "
                              "file does not exist then a new one will be "
                              "created."))
    parser.add_argument(
        "--agent-type",
        choices=["logistic-regression", "mlp-classifier", "sgd-classifier"],
        default="logistic-regression",
        help="The type of machine learning agent to use.",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help=("The number of games to play before selecting the best ones for "
              "training"),
    )
    parser.add_argument(
        "--elite-percentile",
        type=float,
        default=50,
        help=("The quality that a game must have (in terms of score) to be "
              "selected for training."),
    )
    parser.add_argument(
        "--reward-time-multiplier",
        type=float,
        default=1.0,
        help=("The quality that a game must have (in terms of score) to be "
              "selected for training."),
    )
    parser.add_argument(
        "--reward-bounces-multiplier",
        type=float,
        default=0.0,
        help=("The quality that a game must have (in terms of score) to be "
              "selected for training."),
    )
    parser.add_argument(
        "--reward-height-multiplier",
        type=float,
        default=0.0,
        help=("The quality that a game must have (in terms of score) to be "
              "selected for training."),
    )
    parser.add_argument(
        "--punish-moves-multiplier",
        type=float,
        default=1.0,
        help=("The quality that a game must have (in terms of score) to be "
              "selected for training."),
    )
    args = parser.parse_args()

    if args.model_file and os.path.exists(args.model_file):
        m = pickle.load(open(args.model_file, "rb"))
        if not isinstance(m, model.Model):
            m = model.Model(m, game_factory.GameFactory())
    else:
        if args.agent_type == "logistic-regression":
            agent = LogisticRegression(solver="lbfgs", multi_class="auto")
        elif args.agent_type == "sgd-classifier":
            agent = SGDClassifier(loss="log", max_iter=1)
        elif args.agent_type == "mlp-classifier":
            agent = MLPClassifier(hidden_layer_sizes=(20, 20),
                                  warm_start=True,
                                  max_iter=1)
        agent.fit(
            [
                np.array(
                    [0.5 for _ in range(well_bouncer_game.Game.NUM_STATES)])
            ] * well_bouncer_game.Game.NUM_ACTIONS,
            list(range(well_bouncer_game.Game.NUM_ACTIONS)),
        )

        gf = game_factory.GameFactory(
            reward_time_multiplier=args.reward_time_multiplier,
            reward_bounces_multiplier=args.reward_bounces_multiplier,
            reward_height_multiplier=args.reward_height_multiplier,
            punish_moves_multiplier=args.punish_moves_multiplier)

        m = model.Model(agent, gf)
    print("{:>10} {:>9.0f}% {:>10}".format("MEAN", args.elite_percentile,
                                           "KEPT"))
    print("{:>10} {:>10} {:>10}".format("====", "===", "===="))
    while not stop:
        _self_train_once(m.agent, m.game_factory, args.num_games,
                         args.elite_percentile, lambda: stop)
        pickle.dump(m, open(args.model_file, "wb"))


if __name__ == "__main__":
    main()
