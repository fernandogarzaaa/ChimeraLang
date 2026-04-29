"""chimera_runtime — runtime support for ChimeraLang-generated PyTorch code."""
from chimera_runtime.belief_tracker import BeliefTracker
from chimera_runtime.guard_layer import GuardLayer, GuardResult
from chimera_runtime.constitution_layer import ConstitutionLayer, ConstitutionResult
from chimera_runtime.vector_store import VectorStore
from chimera_runtime.spike_runtime import (
    LIFConfig,
    LIFLayer,
    PoissonEncoder,
    RateDecoder,
    STDPDelta,
    STDPDeltaConfig,
    STDPUpdater,
    SpikeTrain,
    lif_simulation,
)
from chimera_runtime.swarm_protocol import (
    Agent,
    GossipMessage,
    SwarmConfig,
    SwarmProtocol,
)
from chimera_runtime.roadmap_systems import (
    CausalModel,
    DifferentialPrivacyEngine,
    MetaLearner,
    PredictiveCodingRuntime,
    ReplayBuffer,
    RewardSystem,
    SelfImprover,
)

__all__ = ["BeliefTracker", "GuardLayer", "GuardResult",
           "ConstitutionLayer", "ConstitutionResult", "VectorStore",
           "LIFConfig", "LIFLayer", "PoissonEncoder", "RateDecoder",
           "STDPDelta", "STDPDeltaConfig", "STDPUpdater", "SpikeTrain",
           "lif_simulation", "Agent", "GossipMessage", "SwarmConfig",
           "SwarmProtocol", "CausalModel", "DifferentialPrivacyEngine",
           "MetaLearner", "PredictiveCodingRuntime", "ReplayBuffer",
           "RewardSystem", "SelfImprover"]
__version__ = "0.1.0"
