import os
import random
import sys
from argparse import ArgumentParser, Namespace
from collections import deque
from datetime import datetime
from pathlib import Path
from pprint import pprint

import numpy as np
import psutil
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from torch.utils.tensorboard import SummaryWriter

from reinforcement_learning.ppo.ppo_agent import PPOAgent

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from utils.timer import Timer
from utils.observation_utils import normalize_observation
from utils.fast_tree_obs import FastTreeObs

try:
    import wandb

    wandb.init(sync_tensorboard=True)
except ImportError:
    print("Install wandb to log to Weights & Biases")

"""
This file shows how to train multiple agents using a reinforcement learning approach.
After training an agent, you can submit it straight away to the NeurIPS 2020 Flatland challenge!

Agent documentation: https://flatland.aicrowd.com/getting-started/rl/multi-agent.html
Submission documentation: https://flatland.aicrowd.com/getting-started/first-submission.html
"""


def create_rail_env(env_params, tree_observation):
    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    max_rails_between_cities = env_params.max_rails_between_cities
    max_rails_in_city = env_params.max_rails_in_city
    seed = env_params.seed

    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=env_params.malfunction_rate,
        min_duration=20,
        max_duration=50
    )

    return RailEnv(
        width=x_dim, height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city
        ),
        schedule_generator=sparse_schedule_generator(),
        number_of_agents=n_agents,
        malfunction_generator_and_process_data=malfunction_from_params(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=seed
    )


def train_agent(train_params, train_env_params, eval_env_params, obs_params):
    # Environment parameters
    n_agents = train_env_params.n_agents
    x_dim = train_env_params.x_dim
    y_dim = train_env_params.y_dim
    n_cities = train_env_params.n_cities
    max_rails_between_cities = train_env_params.max_rails_between_cities
    max_rails_in_city = train_env_params.max_rails_in_city
    seed = train_env_params.seed

    # Unique ID for this training
    now = datetime.now()
    training_id = now.strftime('%y%m%d%H%M%S')

    # Observation parameters
    observation_tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius
    observation_max_path_depth = obs_params.observation_max_path_depth

    # Training parameters
    eps_start = train_params.eps_start
    eps_end = train_params.eps_end
    eps_decay = train_params.eps_decay
    n_episodes = train_params.n_episodes
    checkpoint_interval = train_params.checkpoint_interval
    n_eval_episodes = train_params.n_evaluation_episodes
    restore_replay_buffer = train_params.restore_replay_buffer
    save_replay_buffer = train_params.save_replay_buffer

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    if not train_params.use_fast_tree_observation:
        print("\nUsing standard TreeObs")

        def check_is_observation_valid(observation):
            return observation

        def get_normalized_observation(observation, tree_depth: int, observation_radius=0):
            return normalize_observation(observation, tree_depth, observation_radius)

        tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)
        tree_observation.check_is_observation_valid = check_is_observation_valid
        tree_observation.get_normalized_observation = get_normalized_observation
    else:
        print("\nUsing FastTreeObs")

        def check_is_observation_valid(observation):
            return True

        def get_normalized_observation(observation, tree_depth: int, observation_radius=0):
            return observation

        tree_observation = FastTreeObs(max_depth=observation_tree_depth)
        tree_observation.check_is_observation_valid = check_is_observation_valid
        tree_observation.get_normalized_observation = get_normalized_observation

    # Setup the environments
    train_env = create_rail_env(train_env_params, tree_observation)
    train_env.reset(regenerate_schedule=True, regenerate_rail=True)
    eval_env = create_rail_env(eval_env_params, tree_observation)
    eval_env.reset(regenerate_schedule=True, regenerate_rail=True)

    if not train_params.use_fast_tree_observation:
        # Calculate the state size given the depth of the tree observation and the number of features
        n_features_per_node = train_env.obs_builder.observation_dim
        n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
        state_size = n_features_per_node * n_nodes
    else:
        # Calculate the state size given the depth of the tree observation and the number of features
        state_size = tree_observation.observation_dim

    # Setup renderer
    if train_params.render:
        env_renderer = RenderTool(train_env, gl="PGL")

    # The action space of flatland is 5 discrete actions
    action_size = 5

    action_count = [0] * action_size
    action_dict = dict()
    agent_obs = [None] * n_agents
    agent_prev_obs = [None] * n_agents
    agent_prev_action = [2] * n_agents
    update_values = [False] * n_agents

    # Smoothed values used as target for hyperparameter tuning
    smoothed_eval_normalized_score = -1.0
    smoothed_eval_completion = 0.0

    scores_window = deque(maxlen=checkpoint_interval)  # todo smooth when rendering instead
    completion_window = deque(maxlen=checkpoint_interval)

    # Double Dueling DQN policy
    # policy = DDDQNPolicy(state_size, action_size, train_params)
    policy = PPOAgent(state_size, action_size, n_agents)
    # Load existing policy
    if train_params.load_policy is not "":
        policy.load(train_params.load_policy)

    # Loads existing replay buffer
    if restore_replay_buffer:
        try:
            policy.load_replay_buffer(restore_replay_buffer)
            policy.test()
        except RuntimeError as e:
            print("\n🛑 Could't load replay buffer, were the experiences generated using the same tree depth?")
            print(e)
            exit(1)

    print("\n💾 Replay buffer status: {}/{} experiences".format(len(policy.memory.memory), train_params.buffer_size))

    hdd = psutil.disk_usage('/')
    if save_replay_buffer and (hdd.free / (2 ** 30)) < 500.0:
        print(
            "⚠️  Careful! Saving replay buffers will quickly consume a lot of disk space. You have {:.2f}gb left.".format(
                hdd.free / (2 ** 30)))

    # TensorBoard writer
    writer = SummaryWriter()
    writer.add_hparams(vars(train_params), {})
    writer.add_hparams(vars(train_env_params), {})
    writer.add_hparams(vars(obs_params), {})

    training_timer = Timer()
    training_timer.start()

    print(
        "\n🚉 Training {} trains on {}x{} grid for {} episodes, evaluating on {} episodes every {} episodes. Training id '{}'.\n".format(
            train_env.get_num_agents(),
            x_dim, y_dim,
            n_episodes,
            n_eval_episodes,
            checkpoint_interval,
            training_id
        ))

    for episode_idx in range(n_episodes + 1):
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()
        preproc_timer = Timer()
        inference_timer = Timer()

        # Reset environment
        reset_timer.start()
        train_env_params.n_agents = episode_idx % n_agents + 1
        train_env = create_rail_env(train_env_params, tree_observation)
        obs, info = train_env.reset(regenerate_rail=True, regenerate_schedule=True)
        policy.reset()
        reset_timer.end()

        if train_params.render:
            env_renderer.set_new_rail()

        score = 0
        nb_steps = 0
        actions_taken = []

        # Build initial agent-specific observations
        for agent in train_env.get_agent_handles():
            if tree_observation.check_is_observation_valid(obs[agent]):
                agent_obs[agent] = tree_observation.get_normalized_observation(obs[agent], observation_tree_depth,
                                                                               observation_radius=observation_radius)
                agent_prev_obs[agent] = agent_obs[agent].copy()

        # Max number of steps per episode
        # This is the official formula used during evaluations
        # See details in flatland.envs.schedule_generators.sparse_schedule_generator
        # max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))
        max_steps = train_env._max_episode_steps

        # Run episode
        for step in range(max_steps - 1):
            inference_timer.start()
            policy.start_step()
            for agent in train_env.get_agent_handles():
                if info['action_required'][agent]:
                    update_values[agent] = True

                    action = policy.act(agent_obs[agent], eps=eps_start)

                    action_count[action] += 1
                    actions_taken.append(action)
                else:
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if is currently malfunctioning.
                    update_values[agent] = False
                    action = 0
                action_dict.update({agent: action})
            policy.end_step()
            inference_timer.end()

            # Environment step
            step_timer.start()
            next_obs, all_rewards, done, info = train_env.step(action_dict)
            step_timer.end()

            # Render an episode at some interval
            if train_params.render and episode_idx % checkpoint_interval == 0:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

            # Update replay buffer and train agent
            for agent in train_env.get_agent_handles():
                if update_values[agent] or done['__all__']:
                    # Only learn from timesteps where somethings happened
                    learn_timer.start()
                    policy.step(agent,
                                agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent],
                                agent_obs[agent],
                                done[agent])
                    learn_timer.end()

                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                # Preprocess the new observations
                if tree_observation.check_is_observation_valid(next_obs[agent]):
                    preproc_timer.start()
                    agent_obs[agent] = tree_observation.get_normalized_observation(next_obs[agent],
                                                                                   observation_tree_depth,
                                                                                   observation_radius=observation_radius)
                    preproc_timer.end()

                score += all_rewards[agent]

            nb_steps = step

            if done['__all__']:
                break

        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Collect information about training
        tasks_finished = sum(done[idx] for idx in train_env.get_agent_handles())
        completion = tasks_finished / max(1, train_env.get_num_agents())
        normalized_score = score / (max_steps * train_env.get_num_agents())
        action_probs = action_count / max(1, np.sum(action_count))

        scores_window.append(normalized_score)
        completion_window.append(completion)
        smoothed_normalized_score = np.mean(scores_window)
        smoothed_completion = np.mean(completion_window)

        # Print logs
        if episode_idx % checkpoint_interval == 0:
            policy.save('./checkpoints/' + training_id + '-' + str(episode_idx) + '.pth')

            if save_replay_buffer:
                policy.save_replay_buffer('./replay_buffers/' + training_id + '-' + str(episode_idx) + '.pkl')

            if train_params.render:
                env_renderer.close_window()

            # reset action count
            action_count = [0] * action_size

        print(
            '\r🚂 Episode {}'
            '\t 🏆 Score: {:7.3f}'
            ' Avg: {:7.3f}'
            '\t 💯 Done: {:6.2f}%'
            ' Avg: {:6.2f}%'
            '\t 🎲 Epsilon: {:.3f} '
            '\t 🔀 Action Probs: {}'.format(
                episode_idx,
                normalized_score,
                smoothed_normalized_score,
                100 * completion,
                100 * smoothed_completion,
                eps_start,
                format_action_prob(action_probs)
            ), end=" ")

        # Evaluate policy and log results at some interval
        if episode_idx % checkpoint_interval == 0 and n_eval_episodes > 0:
            scores, completions, nb_steps_eval = eval_policy(eval_env,
                                                             tree_observation,
                                                             policy,
                                                             train_params,
                                                             obs_params)

            writer.add_scalar("evaluation/scores_min", np.min(scores), episode_idx)
            writer.add_scalar("evaluation/scores_max", np.max(scores), episode_idx)
            writer.add_scalar("evaluation/scores_mean", np.mean(scores), episode_idx)
            writer.add_scalar("evaluation/scores_std", np.std(scores), episode_idx)
            writer.add_histogram("evaluation/scores", np.array(scores), episode_idx)
            writer.add_scalar("evaluation/completions_min", np.min(completions), episode_idx)
            writer.add_scalar("evaluation/completions_max", np.max(completions), episode_idx)
            writer.add_scalar("evaluation/completions_mean", np.mean(completions), episode_idx)
            writer.add_scalar("evaluation/completions_std", np.std(completions), episode_idx)
            writer.add_histogram("evaluation/completions", np.array(completions), episode_idx)
            writer.add_scalar("evaluation/nb_steps_min", np.min(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_max", np.max(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_mean", np.mean(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_std", np.std(nb_steps_eval), episode_idx)
            writer.add_histogram("evaluation/nb_steps", np.array(nb_steps_eval), episode_idx)

            smoothing = 0.9
            smoothed_eval_normalized_score = smoothed_eval_normalized_score * smoothing + np.mean(scores) * (
                    1.0 - smoothing)
            smoothed_eval_completion = smoothed_eval_completion * smoothing + np.mean(completions) * (1.0 - smoothing)
            writer.add_scalar("evaluation/smoothed_score", smoothed_eval_normalized_score, episode_idx)
            writer.add_scalar("evaluation/smoothed_completion", smoothed_eval_completion, episode_idx)

        # Save logs to tensorboard
        writer.add_scalar("training/score", normalized_score, episode_idx)
        writer.add_scalar("training/smoothed_score", smoothed_normalized_score, episode_idx)
        writer.add_scalar("training/completion", np.mean(completion), episode_idx)
        writer.add_scalar("training/smoothed_completion", np.mean(smoothed_completion), episode_idx)
        writer.add_scalar("training/nb_steps", nb_steps, episode_idx)
        writer.add_histogram("actions/distribution", np.array(actions_taken), episode_idx)
        writer.add_scalar("actions/nothing", action_probs[RailEnvActions.DO_NOTHING], episode_idx)
        writer.add_scalar("actions/left", action_probs[RailEnvActions.MOVE_LEFT], episode_idx)
        writer.add_scalar("actions/forward", action_probs[RailEnvActions.MOVE_FORWARD], episode_idx)
        writer.add_scalar("actions/right", action_probs[RailEnvActions.MOVE_RIGHT], episode_idx)
        writer.add_scalar("actions/stop", action_probs[RailEnvActions.STOP_MOVING], episode_idx)
        writer.add_scalar("training/epsilon", eps_start, episode_idx)
        writer.add_scalar("training/buffer_size", len(policy.memory), episode_idx)
        writer.add_scalar("training/loss", policy.loss, episode_idx)
        writer.add_scalar("timer/reset", reset_timer.get(), episode_idx)
        writer.add_scalar("timer/step", step_timer.get(), episode_idx)
        writer.add_scalar("timer/learn", learn_timer.get(), episode_idx)
        writer.add_scalar("timer/preproc", preproc_timer.get(), episode_idx)
        writer.add_scalar("timer/total", training_timer.get_current(), episode_idx)


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


def eval_policy(env, tree_observation, policy, train_params, obs_params):
    n_eval_episodes = train_params.n_evaluation_episodes
    max_steps = env._max_episode_steps
    tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius

    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []

    for episode_idx in range(n_eval_episodes):
        agent_obs = [None] * env.get_num_agents()
        score = 0.0

        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        final_step = 0

        for step in range(max_steps - 1):
            policy.start_step()
            for agent in env.get_agent_handles():
                if tree_observation.check_is_observation_valid(agent_obs[agent]):
                    agent_obs[agent] = tree_observation.get_normalized_observation(obs[agent], tree_depth=tree_depth,
                                                                                   observation_radius=observation_radius)

                action = 0
                if info['action_required'][agent]:
                    if tree_observation.check_is_observation_valid(agent_obs[agent]):
                        action = policy.act(agent_obs[agent], eps=0.0)
                action_dict.update({agent: action})
            policy.end_step()
            obs, all_rewards, done, info = env.step(action_dict)

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if done['__all__']:
                break

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

    print("\t✅ Eval: score {:.3f} done {:.1f}%".format(np.mean(scores), np.mean(completions) * 100.0))

    return scores, completions, nb_steps


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--n_episodes", help="number of episodes to run", default=5400, type=int)
    parser.add_argument("-t", "--training_env_config", help="training config id (eg 0 for Test_0)", default=1, type=int)
    parser.add_argument("-e", "--evaluation_env_config", help="evaluation config id (eg 0 for Test_0)", default=0,
                        type=int)
    parser.add_argument("--n_evaluation_episodes", help="number of evaluation episodes", default=5, type=int)
    parser.add_argument("--checkpoint_interval", help="checkpoint interval", default=100, type=int)
    parser.add_argument("--eps_start", help="max exploration", default=1.0, type=float)
    parser.add_argument("--eps_end", help="min exploration", default=0.01, type=float)
    parser.add_argument("--eps_decay", help="exploration decay", default=0.998, type=float)
    parser.add_argument("--buffer_size", help="replay buffer size", default=int(1e7), type=int)
    parser.add_argument("--buffer_min_size", help="min buffer size to start training", default=0, type=int)
    parser.add_argument("--restore_replay_buffer", help="replay buffer to restore", default="", type=str)
    parser.add_argument("--save_replay_buffer", help="save replay buffer at each evaluation interval", default=False,
                        type=bool)
    parser.add_argument("--batch_size", help="minibatch size", default=128, type=int)
    parser.add_argument("--gamma", help="discount factor", default=0.99, type=float)
    parser.add_argument("--tau", help="soft update of target parameters", default=1e-3, type=float)
    parser.add_argument("--learning_rate", help="learning rate", default=0.5e-4, type=float)
    parser.add_argument("--hidden_size", help="hidden size (2 fc layers)", default=128, type=int)
    parser.add_argument("--update_every", help="how often to update the network", default=8, type=int)
    parser.add_argument("--use_gpu", help="use GPU if available", default=False, type=bool)
    parser.add_argument("--num_threads", help="number of threads PyTorch can use", default=1, type=int)
    parser.add_argument("--render", help="render 1 episode in 100", action='store_true')
    parser.add_argument("--load_policy", help="policy filename (reference) to load", default="", type=str)
    parser.add_argument("--use_fast_tree_observation", help="use FastTreeObs instead of stock TreeObs",
                        action='store_true')
    parser.add_argument("--max_depth", help="max depth", default=1, type=int)

    training_params = parser.parse_args()
    env_params = [
        {
            # Test_0
            "n_agents": 5,
            "x_dim": 25,
            "y_dim": 25,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 50,
            "seed": 0
        },
        {
            # Test_1
            "n_agents": 10,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 100,
            "seed": 0
        },
        {
            # Test_2
            "n_agents": 20,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 3,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
    ]

    obs_params = {
        "observation_tree_depth": training_params.max_depth,
        "observation_radius": 10,
        "observation_max_path_depth": 30
    }


    def check_env_config(id):
        if id >= len(env_params) or id < 0:
            print("\n🛑 Invalid environment configuration, only Test_0 to Test_{} are supported.".format(
                len(env_params) - 1))
            exit(1)


    check_env_config(training_params.training_env_config)
    check_env_config(training_params.evaluation_env_config)

    training_env_params = env_params[training_params.training_env_config]
    evaluation_env_params = env_params[training_params.evaluation_env_config]

    # FIXME hard-coded for sweep search
    # see https://wb-forum.slack.com/archives/CL4V2QE59/p1602931982236600 to implement properly
    # training_params.use_fast_tree_observation = True

    print("\nTraining parameters:")
    pprint(vars(training_params))
    print("\nTraining environment parameters (Test_{}):".format(training_params.training_env_config))
    pprint(training_env_params)
    print("\nEvaluation environment parameters (Test_{}):".format(training_params.evaluation_env_config))
    pprint(evaluation_env_params)
    print("\nObservation parameters:")
    pprint(obs_params)

    os.environ["OMP_NUM_THREADS"] = str(training_params.num_threads)
    train_agent(training_params, Namespace(**training_env_params), Namespace(**evaluation_env_params),
                Namespace(**obs_params))
