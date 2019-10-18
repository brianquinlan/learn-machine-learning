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

import gym
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import sklearn.exceptions
import tqdm

import model


def _make_move(agent, state):
    probs = agent.predict_proba([state])[0]
    a = np.random.choice(agent.classes_, p=probs)
    return a


def _select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    reward_threshold = np.percentile(rewards_batch, percentile)
    elite_reward_indices = [
        i for i in range(len(rewards_batch))
        if rewards_batch[i] > reward_threshold
    ]

    elite_states = []
    elite_actions = []
    elite_scores = []
    for i in elite_reward_indices:
        elite_states.extend(states_batch[i])
        elite_actions.extend(actions_batch[i])
        elite_scores.append(rewards_batch[i])

    return elite_states, elite_actions, elite_scores


def _generate_session(agent, env, state):
    states = []
    actions = []
    total_reward = 0

    while True:
        states.append(state)

        action = _make_move(agent, state)
        state, reward, done, _ = env.step(action)

        actions.append(action)

        total_reward += reward

        if done:
            break
    return states, actions, total_reward


def _train_one_state(agent, game, num_tries, elite_percentile, stop_checker):
    env = gym.make(game)
    initial_state = env.reset()
    cenv = pickle.dumps(env)
    state_lists = []
    action_lists = []
    scores = []
    for _ in tqdm.tqdm(range(num_tries),
                       leave=False,
                       ncols=31,
                       bar_format="{bar}"):
        states, actions, score = _generate_session(agent, pickle.loads(cenv),
                                                   initial_state)
        if stop_checker():
            return [], [], []

        state_lists.append(states)
        action_lists.append(actions)
        scores.append(score)
    elite_states, elite_actions, elite_scores = _select_elites(state_lists, action_lists,
                                                               scores,
                                                               elite_percentile)
    return elite_states, elite_actions, elite_scores


def _self_train_once(agent,
                     game,
                     num_games,
                     elite_percentile, elite_percentile_per_state,
                     stop_checker):
    combined_states = []
    combined_actions = []
    combined_scores  = []
    for _ in tqdm.tqdm(range(num_games),
                       leave=False,
                       ncols=31,
                       bar_format="{bar}"):
        states, actions, scores = _train_one_state(agent, game, 100,
                                                   elite_percentile_per_state,
                                                   stop_checker)
        if stop_checker():
            return
        combined_states.extend(states)
        combined_actions.extend(actions)
        combined_scores.extend(scores)


#    elite_states, elite_actions, elite_scores = _select_elites(state_lists,
#                                                               action_lists,
#                                                               score_lists,
#                                                               elite_percentile)

    print("{:10.1f} {:10}".format(
        np.mean(combined_scores),
        len(combined_states),
    ))

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=sklearn.exceptions.ConvergenceWarning)
        agent.fit(combined_states, combined_actions)


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
        "--elite-percentile-per-state",
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
            m = model.Model(m, 'LunarLander-v2')
    else:
        if args.agent_type == "logistic-regression":
            agent = LogisticRegression(solver="lbfgs", multi_class="auto")
        elif args.agent_type == "sgd-classifier":
            agent = SGDClassifier(loss="log", max_iter=1)
        elif args.agent_type == "mlp-classifier":
            agent = MLPClassifier(hidden_layer_sizes=(10, 10, 10, 10),
                                  warm_start=True)
#                                  max_iter=1)
        agent.fit(
            [
                np.array(
                    [0.5 for _ in range(8)])
            ] * 4,
            list(range(4)),
        )

        m = model.Model(agent, 'LunarLander-v2')
    print("{:>10} {:>9.0f}% {:>10}".format("MEAN", args.elite_percentile,
                                           "KEPT"))
    print("{:>10} {:>10} {:>10}".format("====", "===", "===="))
    while not stop:
        _self_train_once(m.agent, m.game, args.num_games,
                         args.elite_percentile, 90, lambda: stop)
        pickle.dump(m, open(args.model_file, "wb"))


if __name__ == "__main__":
    main()
