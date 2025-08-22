import numpy as np
import random


class DreamEnv:
    """
    A simple queueing environment with support for failures and recovery.
    Observations: [queue_length, server_status]
    Actions: 0 = do nothing, 1 = add job
    Rewards: throughput - drops - penalty for downtime
    """

    def __init__(self, max_queue=10, fail_prob=0.05, repair_prob=0.2):
        self.max_queue = max_queue
        self.fail_prob = fail_prob
        self.repair_prob = repair_prob

        # Define action and observation spaces
        self.action_space = [0, 1]      # two discrete actions
        self.observation_space = (2,)      # observation = [queue_length, server_status]

        # New attributes to track downtime for MTTR
        self.server_down_time = []
        self.downtime_start_step = None

        self.reset()

    def reset(self):
        self.queue_len = 0
        self.server_up = True
        self.jobs_processed = 0
        self.jobs_dropped = 0
        self.steps = 0
        # Reset downtime tracking
        self.downtime_start_step = None
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.queue_len, int(self.server_up)], dtype=np.float32)

    def step(self, action: int):
        """
        action = 0 -> do nothing
        action = 1 -> add one job
        """
        reward = 0

        # New job arrives if action == 1
        if action == 1:
            if self.queue_len < self.max_queue:
                self.queue_len += 1
            else:
                # Queue overflow → job dropped
                self.jobs_dropped += 1
                reward -= 1

        # Track server failure and recovery for MTTR calculation
        prev_server_status = self.server_up

        # Server may fail randomly
        if self.server_up and random.random() < self.fail_prob:
            self.server_up = False
        
        # Server may recover randomly
        if not self.server_up and random.random() < self.repair_prob:
            self.server_up = True

        # Check for status changes and update downtime tracking
        if prev_server_status and not self.server_up:
            # Server just failed
            self.downtime_start_step = self.steps
        elif not prev_server_status and self.server_up and self.downtime_start_step is not None:
            # Server just recovered
            downtime_duration = self.steps - self.downtime_start_step
            self.server_down_time.append(downtime_duration)
            self.downtime_start_step = None


        # Process job if server is up
        if self.server_up and self.queue_len > 0:
            self.queue_len -= 1
            self.jobs_processed += 1
            reward += 1

        self.steps += 1
        done = self.steps >= 50  # episode length fixed

        return self._get_obs(), reward, done, {}

    def render(self):
        print(
            f"[Step {self.steps}] Queue={self.queue_len}, Server={'UP' if self.server_up else 'DOWN'}, "
            f"Processed={self.jobs_processed}, Dropped={self.jobs_dropped}"
        )