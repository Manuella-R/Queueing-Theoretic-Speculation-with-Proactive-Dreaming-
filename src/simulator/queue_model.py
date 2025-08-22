import numpy as np

class SpeculativeServer:
    """
    Single-server queue with speculative timeout τ. 
    Service times ~ Exp(mu). Speculation modeled as: if service exceeds τ, a duplicate is launched;
    effective completion is min(original residual, new sample). This changes the effective service distribution.
    We track utilization ρ_τ and compute an idle budget b = β * max(0, 1 - ρ_τ).
    This is a simplified model for research scaffolding.
    """
    def __init__(self, mu: float, lam: float, tau: float, beta: float=0.2, seed: int=0):
        assert mu > 0 and lam >= 0 and tau >= 0
        self.mu = mu      # service rate
        self.lam = lam    # arrival rate
        self.tau = tau    # speculation timeout
        self.beta = beta  # idle-budget scaling
        self.rng = np.random.default_rng(seed)

        # state
        self.queue = []
        self.time = 0.0
        self.busy_until = 0.0
        self.latencies = []
        self.completions = 0
        self.failures = []  # times of injected failures (for MTTR calc)
        self.recoveries = []

        # background (dream) accounting
        self.dream_cpu_secs = 0.0
        self.dream_events = 0
        self.last_failure_end = None

    def draw_service(self):
        return self.rng.exponential(1.0/self.mu)

    def effective_service_with_speculation(self, s):
        """
        If s <= τ, no speculation. Else, at τ launch a duplicate with fresh Exp(mu).
        Remaining time of original after τ is (s-τ). Completion is τ + min(s-τ, s2).
        Expected value used for utilization estimate; here we simulate sample-wise.
        """
        if s <= self.tau:
            return s, False
        else:
            # duplicate
            s2 = self.draw_service()
            rem = s - self.tau
            return self.tau + min(rem, s2), True

    def estimate_rho_tau(self, samples=2000):
        # Monte Carlo estimate of E[S_tau] to compute utilization ρ_τ = λ E[S_τ]
        ss = self.rng.exponential(1.0/self.mu, size=samples)
        eff = []
        for s in ss:
            e, _ = self.effective_service_with_speculation(s)
            eff.append(e)
        est = float(np.mean(eff))
        return min(0.999, self.lam * est)

    def idle_budget(self):
        rho = self.estimate_rho_tau()
        return max(0.0, self.beta * (1.0 - rho))

    def step(self, dt: float, inject_failure=False):
        """
        Advance simulated time by dt. During dt, we process foreground work first.
        Any leftover idle fraction is available to run background dreams, bounded by budget.
        """
        # Foreground: approximate fluid processing at rate 1 while busy
        # Generate arrivals in dt (Poisson)
        arrivals = self.rng.poisson(self.lam * dt)
        for _ in range(arrivals):
            s = self.draw_service()
            eff_s, _ = self.effective_service_with_speculation(s)
            self.queue.append(eff_s)

        # Process queue
        proc_time = dt
        while proc_time > 1e-12 and self.queue:
            job = self.queue[0]
            if job <= proc_time:
                proc_time -= job
                self.latencies.append(job)
                self.completions += 1
                self.queue.pop(0)
            else:
                self.queue[0] = job - proc_time
                proc_time = 0.0

        # Background dreams: use remaining proc_time as idle; bounded by budget
        idle = proc_time
        budget = self.idle_budget() * dt  # budget scaled over dt window
        dream_time = min(idle, budget)
        if dream_time > 0:
            self.dream_cpu_secs += dream_time
            self.dream_events += 1

        self.time += dt

    def metrics(self):
        p = np.percentile(self.latencies, [50, 95, 99]) if self.latencies else [np.nan]*3
        return {
            "time": self.time,
            "completions": self.completions,
            "q_len": len(self.queue),
            "lat_p50": float(p[0]),
            "lat_p95": float(p[1]),
            "lat_p99": float(p[2]),
            "rho_tau_est": float(self.estimate_rho_tau()),
            "idle_budget": float(self.idle_budget()),
            "dream_cpu_secs": float(self.dream_cpu_secs),
            "dream_events": int(self.dream_events),
        }
