def idle_budget(beta: float, rho_tau: float) -> float:
    return max(0.0, beta * (1.0 - min(0.999, rho_tau)))
