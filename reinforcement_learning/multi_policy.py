from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnvActions

from reinforcement_learning.dddqn_policy import DDDQNPolicy
from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent


class MultiPolicy(DDDQNPolicy):
    def __init__(self, state_size, action_size, parameters, evaluation_mode=False):
        super().__init__(state_size, action_size, parameters, evaluation_mode)
        self.dead_lock_avoidance_agent = None

    def set_rail_env(self, rail_env):
        self.dead_lock_avoidance_agent = DeadLockAvoidanceAgent(rail_env)
        self.dead_lock_avoidance_agent.reset()

    def act(self, handle, state, eps=0.):
        agent = self.dead_lock_avoidance_agent.env.agents[handle]
        if agent.status < RailAgentStatus.ACTIVE:
            action_dlaa = self.dead_lock_avoidance_agent.act([handle], eps)
            #if action_dlaa == RailEnvActions.STOP_MOVING:
            return action_dlaa
        action = super().act(state, eps)
        return action

    def start_step(self):
        super().start_step()
        self.dead_lock_avoidance_agent.start_step()

    def end_step(self):
        super().end_step()
        self.dead_lock_avoidance_agent.end_step()
