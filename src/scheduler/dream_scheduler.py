from ..simulator.env import DreamEnv
from ..rl.bandit import EpsilonGreedyBandit

class DreamScheduler:
    """
    Ties together the environment and RL policy.
    At each step:
      - Observe state
      - Select a dream (bandit)
      - Step env; rewards reflect benefit-cost tradeoff
    """
    def __init__(self, env: DreamEnv, eps=0.1):
        self.env = env
        self.agent = EpsilonGreedyBandit(n_actions=env.K, eps=eps)

    def run(self, episodes=50, dt=1.0):
        history = []
        state = self.env.reset()
        for t in range(episodes):
            a = self.agent.select()
            next_state, reward, done, metrics = self.env.step(a, dt=dt)
            self.agent.update(a, reward)
            history.append({**metrics, "action": a, "reward": reward})
            state = next_state
        return history
