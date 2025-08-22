import matplotlib.pyplot as plt
from src.simulator.env import DreamEnv
from src.simulator.utils import compute_metrics

# Change the import to use the new QLearningAgent
from src.simulator.q_agent import QLearningAgent


def run_sim(episodes=20):
    env = DreamEnv()
    # Instantiate the QLearningAgent, passing the environment object
    agent = QLearningAgent(env.action_space, env)

    rewards, throughputs, drops = [], [], []

    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward

            # The agent learns after each step
            agent.learn(state, action, reward, next_state)
            state = next_state

        # Decay epsilon at the end of each episode to reduce exploration
        agent.decay_epsilon()
        
        # Compute episode-level metrics after the episode is complete
        info = compute_metrics(env)

        rewards.append(ep_reward)
        throughputs.append(info["throughput"])
        drops.append(info["drop_rate"])

        # Print the progress
        print(f"[Episode {ep+1}/{episodes}] "
              f"Reward={ep_reward:.2f}, "
              f"Throughput={info['throughput']:.3f}, "
              f"DropRate={info['drop_rate']:.3f}, "
              f"Epsilon={agent.epsilon:.3f}") # Print epsilon to monitor decay

    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    axs[0].plot(rewards, marker="o")
    axs[0].set_title("Reward per Episode")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")

    axs[1].plot(throughputs, marker="s", color="green")
    axs[1].set_title("Throughput per Episode")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Throughput")

    axs[2].plot(drops, marker="x", color="red")
    axs[2].set_title("Drop Rate per Episode")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Drop Rate")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_sim(episodes=500)