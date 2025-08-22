import numpy as np

def compute_metrics(env: "DreamEnv"):
    """
    Compute episode-level metrics.
    """
    throughput = env.jobs_processed / max(1, env.steps)
    drop_rate = env.jobs_dropped / max(1, env.steps)

    # Calculate fault avoidance rate
    fault_avoidance_rate = 1 - drop_rate

    return {
        "throughput": throughput,
        "drop_rate": drop_rate,
        "processed": env.jobs_processed,
        "dropped": env.jobs_dropped,
        "fault_avoidance_rate": fault_avoidance_rate,
    }


def compute_mttr(env: "DreamEnv"):
    """
    Compute Mean Time to Recovery (MTTR) based on downtime logs.
    """
    if not env.server_down_time:
        return 0  # No downtime occurred in this episode
    return np.mean(env.server_down_time)