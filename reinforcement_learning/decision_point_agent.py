from flatland.envs.rail_env import RailEnv, RailEnvActions

from reinforcement_learning.policy import HybridPolicy
from utils.agent_action_config import map_rail_env_action, map_action
from utils.agent_can_choose_helper import AgentCanChooseHelper
from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent


class DecisionPointAgent(HybridPolicy):

    def __init__(self, env: RailEnv, state_size, action_size, learning_agent, transfer_learning=False):
        print(">> DecisionPointAgent")
        super(DecisionPointAgent, self).__init__()
        self.env = env

        self.state_size = state_size
        self.action_size = action_size
        self.learning_agent = learning_agent
        self.transfer_learning = transfer_learning

        self.memory = self.learning_agent.memory
        self.loss = self.learning_agent.loss

        self.agent_can_choose_helper = AgentCanChooseHelper()
        self.dead_lock_avoidance_agent = DeadLockAvoidanceAgent(env, action_size, enable_eps=False)

    def step(self, handle, state, action, reward, next_state, done):
        self.learning_agent.step(handle, state, action, reward, next_state, done)
        self.loss = self.learning_agent.loss

    def act(self, handle, state, eps=0.):
        agent = self.env.agents[handle]
        agents_on_switch, \
        agents_near_to_switch, \
        agents_near_to_switch_all, \
        agents_on_switch_all = \
            self.agent_can_choose_helper.check_agent_decision(agent.position, agent.direction)

        if agents_on_switch or agents_near_to_switch:
            action = map_action(self.dead_lock_avoidance_agent.act(handle, state, eps))
            if self.transfer_learning:
                return map_rail_env_action(action)
            if action == RailEnvActions.STOP_MOVING:
                return map_rail_env_action(RailEnvActions.STOP_MOVING)
            if agents_near_to_switch or agents_on_switch or agent.position is None:
                return self.learning_agent.act(handle, state, eps)
        return map_rail_env_action(RailEnvActions.MOVE_FORWARD)

    def save(self, filename):
        self.learning_agent.save(filename)
        self.dead_lock_avoidance_agent.save(filename)

    def load(self, filename):
        self.learning_agent.load(filename)
        self.dead_lock_avoidance_agent.load(filename)

    def start_step(self, train):
        self.learning_agent.start_step(train)
        self.dead_lock_avoidance_agent.start_step(train)

    def end_step(self, train):
        self.learning_agent.end_step(train)
        self.dead_lock_avoidance_agent.end_step(train)

    def start_episode(self, train):
        self.learning_agent.start_episode(train)
        self.dead_lock_avoidance_agent.start_episode(train)

    def end_episode(self, train):
        self.learning_agent.end_episode(train)
        self.dead_lock_avoidance_agent.end_episode(train)

    def load_replay_buffer(self, filename):
        self.learning_agent.load_replay_buffer(filename)
        self.dead_lock_avoidance_agent.load_replay_buffer(filename)

    def test(self):
        self.learning_agent.test()
        self.dead_lock_avoidance_agent.test()

    def reset(self, env: RailEnv):
        self.env = env
        self.learning_agent.reset(env)
        self.agent_can_choose_helper.build_data(self.env)
        self.dead_lock_avoidance_agent.reset(env)

    def clone(self):
        return self
