"""selfimprove: an MLflow harness for self-improving agents.

signal -> store -> update -> gate, for an agent you bring (see .env.example):

    from selfimprove import agents, traces
    traces.setup()
    agents.run("How long do refunds take?", source="production")   # one traced call
"""

from selfimprove import config  # noqa: F401  (loads .env and MLflow env defaults first)

__version__ = "0.1.0"
