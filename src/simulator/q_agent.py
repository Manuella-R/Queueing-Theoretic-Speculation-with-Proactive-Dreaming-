import numpy as np
import random

class QLearningAgent:
    """
    A Q-learning agent for the DreamEnv environment.
    It learns a Q-table to map states to optimal actions.
    """

    def __init__(self, action_space, env, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay_rate=0.995, min_epsilon=0.01):
        """
        Initializes the Q-learning agent with hyperparameters.

        Args:
            action_space (list): A list of possible actions.
            env (DreamEnv): The environment object to get max_queue length.
            learning_rate (float): The learning rate (alpha).
            discount_factor (float): The discount factor (gamma).
            epsilon (float): The initial exploration rate (epsilon).
            epsilon_decay_rate (float): The rate at which epsilon decays.
            min_epsilon (float): The minimum value for epsilon.
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon

        # The state is [queue_length, server_status].
        # queue_length can go from 0 to env.max_queue.
        # server_status is 0 (down) or 1 (up).
        # The Q-table shape will be (max_queue + 1, 2, number_of_actions)
        self.q_table = np.zeros((env.max_queue + 1, 2, len(action_space)))

    def act(self, state):
        """
        Chooses an action based on an epsilon-greedy policy.

        Args:
            state (np.array): The current state of the environment.

        Returns:
            int: The chosen action.
        """
        # Get the integer values for the state
        queue_len, server_status = int(state[0]), int(state[1])

        # Epsilon-greedy exploration vs. exploitation
        if random.random() < self.epsilon:
            # Exploration: choose a random action
            return random.choice(self.action_space)
        else:
            # Exploitation: choose the best action from the Q-table
            # The action is the index of the max Q-value
            return np.argmax(self.q_table[queue_len, server_status])

    def learn(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Bellman equation.

        Args:
            state (np.array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.array): The state after the action.
        """
        # Get integer values for current and next states
        queue_len, server_status = int(state[0]), int(state[1])
        next_queue_len, next_server_status = int(next_state[0]), int(next_state[1])

        # Get the current Q-value
        current_q = self.q_table[queue_len, server_status, action]

        # Get the maximum Q-value for the next state
        max_next_q = np.max(self.q_table[next_queue_len, next_server_status])

        # Calculate the new Q-value using the Q-learning formula
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update the Q-table
        self.q_table[queue_len, server_status, action] = new_q

    def decay_epsilon(self):
        """
        Decays the epsilon value to reduce exploration over time.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)