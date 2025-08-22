# Dream-Queue Hybrid
Hybrid queueing-theoretic speculation + RL-based proactive "dreaming" scaffold.

## What is this?
A minimal, runnable research scaffold to:
- Simulate foreground jobs with speculative execution and compute the safe idle budget.
- Use an RL agent to select "dream" (what-if) simulations that run only on idle capacity.
- Measure baseline vs hybrid on latency and recovery metrics.

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_sim.py --episodes 50 --plot
```
Artifacts (plots/CSV) land in `outputs/`.

## Layout
- `src/simulator/queue_model.py` — M/M/1(ish) simulator with speculative timeout τ, computes ρ_τ and idle budget.
- `src/rl/bandit.py` — simple ε-greedy bandit to pick dreams.
- `src/scheduler/dream_scheduler.py` — runs dreams when budget allows (strict preemptive policy simulated).
- `src/evaluation/metrics.py` — latency percentiles, MTTR from injected failures.
- `scripts/run_sim.py` — end-to-end experiment runner.
- `k8s/manifests/` — example PriorityClass + Job manifests for real clusters (illustrative).
- `tests/` — minimal unit tests for sanity.

> This is a **scaffold**: the math is simplified for clarity. Extend as needed for your study.

