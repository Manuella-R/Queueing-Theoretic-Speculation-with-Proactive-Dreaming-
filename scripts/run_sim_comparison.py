import matplotlib.pyplot as plt
import numpy as np # Import numpy for mean calculation
from src.simulator.env import DreamEnv
from src.simulator.utils import compute_metrics, compute_mttr
from src.simulator.q_agent import QLearningAgent
from src.simulator.reactive_agent import ReactiveAgent


def run_experiment(agent, episodes=500):
    """
    Runs a single simulation experiment with a given agent.
    Returns the collected rewards, throughputs, and drop rates.
    """
    env = DreamEnv()
    rewards, throughputs, drops, mttrs = [], [], [], []

    print(f"--- Running experiment with {agent.__class__.__name__} ---")

    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward

            if hasattr(agent, 'learn'):
                agent.learn(state, action, reward, next_state)
            
            state = next_state

        if hasattr(agent, 'decay_epsilon'):
            agent.decay_epsilon()
        
        info = compute_metrics(env)
        mttr = compute_mttr(env)

        rewards.append(ep_reward)
        throughputs.append(info["throughput"])
        drops.append(info["drop_rate"])
        mttrs.append(mttr)

    print(f"--- Experiment with {agent.__class__.__name__} finished. ---")
    return rewards, throughputs, drops, mttrs


if __name__ == "__main__":
    episodes = 500

    q_agent = QLearningAgent(DreamEnv().action_space, DreamEnv())
    q_rewards, q_throughputs, q_drops, q_mttrs = run_experiment(q_agent, episodes)

    reactive_agent = ReactiveAgent(DreamEnv().action_space)
    reactive_rewards, reactive_throughputs, reactive_drops, reactive_mttrs = run_experiment(reactive_agent, episodes)

    # --- Plotting the comparative graphs ---
    fig, axs = plt.subplots(4, 1, figsize=(10, 16))
    fig.suptitle('Performance Comparison: Q-Learning Agent vs. Reactive Agent', fontsize=16)

    # Reward Comparison
    axs[0].plot(q_rewards, marker="o", linestyle='-', label="Q-Learning Agent")
    axs[0].plot(reactive_rewards, marker="s", linestyle='--', label="Reactive Agent")
    axs[0].set_title("Reward per Episode")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")
    axs[0].legend()
    axs[0].grid(True, linestyle='--')

    # Throughput Comparison
    axs[1].plot(q_throughputs, marker="o", linestyle='-', color="green", label="Q-Learning Agent")
    axs[1].plot(reactive_throughputs, marker="s", linestyle='--', color="lime", label="Reactive Agent")
    axs[1].set_title("Throughput per Episode")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Throughput")
    axs[1].legend()
    axs[1].grid(True, linestyle='--')

    # Drop Rate Comparison
    axs[2].plot(q_drops, marker="x", linestyle='-', color="red", label="Q-Learning Agent")
    axs[2].plot(reactive_drops, marker="^", linestyle='--', color="orange", label="Reactive Agent")
    axs[2].set_title("Drop Rate per Episode")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Drop Rate")
    axs[2].legend()
    axs[2].grid(True, linestyle='--')

    # MTTR Comparison (New Plot)
    axs[3].plot(q_mttrs, marker="v", linestyle='-', color="purple", label="Q-Learning Agent")
    axs[3].plot(reactive_mttrs, marker="d", linestyle='--', color="magenta", label="Reactive Agent")
    axs[3].set_title("Mean Time to Recovery (MTTR) per Episode")
    axs[3].set_xlabel("Episode")
    axs[3].set_ylabel("MTTR (steps)")
    axs[3].legend()
    axs[3].grid(True, linestyle='--')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()