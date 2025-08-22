import random

class ReactiveAgent:
    """
    A reactive agent that makes decisions based on the current state only,
    without any learning or foresight. It represents a traditional speculative
    mechanism that only responds to immediate system conditions.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        """
        Chooses an action based on a simple reactive policy.
        
        Args:
            state (np.array): The current state, [queue_length, server_status].

        Returns:
            int: The chosen action (0 or 1).
        """
        queue_len, server_status = state[0], state[1]

        # Check if the server is up and if the queue is not full.
        # This is the "reactive" logic: add a job if conditions are good.
        if server_status == 1 and queue_len < 10:  # Assuming max_queue is 10 from env.py
            return 1 # Add a job
        else:
            return 0 # Do nothing