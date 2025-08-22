from src.simulator.queue_model import SpeculativeServer

def test_idle_budget_increases_when_load_decreases():
    s1 = SpeculativeServer(mu=1.0, lam=0.8, tau=0.5, beta=0.3, seed=1)
    b1 = s1.idle_budget()
    s2 = SpeculativeServer(mu=1.0, lam=0.4, tau=0.5, beta=0.3, seed=1)
    b2 = s2.idle_budget()
    assert b2 >= b1
