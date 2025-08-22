from dataclasses import dataclass

@dataclass
class FaultScenario:
    name: str
    duration: float  # seconds
    severity: float  # scale 0..1 (affects mu or lam in a higher-fidelity model)

# Minimal placeholder library of "dreams"
SCENARIOS = [
    FaultScenario("node_crash_short", duration=1.0, severity=0.7),
    FaultScenario("bandwidth_drop", duration=2.0, severity=0.4),
    FaultScenario("gc_pause", duration=0.5, severity=0.5),
    FaultScenario("load_spike", duration=1.5, severity=0.9),
]
