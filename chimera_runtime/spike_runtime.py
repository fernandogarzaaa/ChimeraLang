"""Spiking neural network runtime — LIF neurons, STDP, spike encoding/decoding."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LIFState:
    """Leaky Integrate-and-Fire neuron membrane state."""
    membrane: float = 0.0
    synaptic: float = 0.0
    spiked: bool = False
    spike_history: list = field(default_factory=list)


@dataclass
class LIFConfig:
    """Configuration for a Leaky Integrate-and-Fire layer."""
    tau_mem: float = 20.0       # membrane time constant (ms)
    tau_syn: float = 5.0         # synaptic time constant (ms)
    threshold: float = 1.0       # firing threshold
    reset: str = "subtract"      # "subtract" | "zero"
    dt: float = 1.0             # simulation timestep (ms)


@dataclass
class STDPDelta:
    """STDP weight update delta."""
    weight_delta: float
    pre_spike: bool
    post_spike: bool
    timing_diff: float  # ms between pre and post spike


@dataclass
class STDPDeltaConfig:
    """STDP learning rule configuration."""
    a_plus: float = 0.01   # potentiation learning rate
    a_minus: float = 0.012  # depression learning rate
    tau_plus: float = 20.0  # potentiation time constant (ms)
    tau_minus: float = 20.0  # depression time constant (ms)
    w_min: float = 0.0
    w_max: float = 1.0


class LIFLayer:
    """Leaky Integrate-and-Fire layer — vectorized simulation."""

    def __init__(self, n_neurons: int, config: LIFConfig | None = None):
        self.n_neurons = n_neurons
        self.config = config or LIFConfig()
        self.weights: list[list[float]] = []
        self._states: list[LIFState] = []

    def initialize_weights(self, input_dim: int) -> None:
        """Initialize random weights from input to this layer."""
        scale = 1.0 / math.sqrt(input_dim)
        self.weights = [
            [float((hash((i, j)) % 1000) / 1000 * 2 * scale - scale)
             for j in range(input_dim)]
            for i in range(self.n_neurons)
        ]
        self._states = [LIFState() for _ in range(self.n_neurons)]

    def forward_step(self, inputs: list[float], timestep: float) -> list[bool]:
        """Single simulation step across all neurons.

        Args:
            inputs: input current for each neuron
            timestep: current simulation time in ms

        Returns:
            list of bool — which neurons spiked this step
        """
        cfg = self.config
        spikes = []
        new_states = []

        for i, state in enumerate(self._states):
            # Synaptic current decay
            state.synaptic *= math.exp(-cfg.dt / cfg.tau_syn)

            # Membrane potential integration
            state.membrane += state.synaptic
            state.membrane += inputs[i]

            # Leaky decay of membrane
            state.membrane *= math.exp(-cfg.dt / cfg.tau_mem)

            # Check spike
            if state.membrane >= cfg.threshold:
                spikes.append(True)
                state.spike_history.append(timestep)
                if cfg.reset == "subtract":
                    state.membrane -= cfg.threshold
                else:
                    state.membrane = 0.0
            else:
                spikes.append(False)

            new_states.append(state)

        self._states = new_states
        return spikes

    def forward_timesteps(self, inputs: list[list[float]], timesteps: int) -> list[list[bool]]:
        """Run simulation for multiple timesteps.

        Args:
            inputs: list of input vectors, one per timestep
            timesteps: number of simulation steps

        Returns:
            spike_train: list of lists of bool — shape [timesteps, n_neurons]
        """
        if len(self.weights) == 0:
            input_dim = len(inputs[0]) if inputs else 0
            self.initialize_weights(input_dim)

        spike_train = []
        for t in range(timesteps):
            inp = inputs[t] if t < len(inputs) else [0.0] * self.n_neurons
            spikes = self.forward_step(inp, float(t * self.config.dt))
            spike_train.append(spikes)

        return spike_train

    def get_spike_train(self) -> list[list[bool]]:
        """Return the full spike train across all timesteps."""
        return [[s.spiked for s in self._states] for _ in range(len(self._states[0].spike_history) // self.n_neurons if self._states else 0)]


class STDPUpdater:
    """Spike-Timing Dependent Plasticity weight updater."""

    def __init__(self, config: STDPDeltaConfig | None = None):
        self.config = config or STDPDeltaConfig()

    def compute_delta(self, pre_spike_time: float, post_spike_time: float) -> STDPDelta:
        """Compute STDP weight change given spike timings.

        Args:
            pre_spike_time: time of presynaptic spike
            post_spike_time: time of postsynaptic spike

        Returns:
            STDPDelta with weight change
        """
        cfg = self.config
        dt = post_spike_time - pre_spike_time  # positive = pre before post (causal)

        if dt > 0:
            # Potentiation: pre fires BEFORE post → strengthen
            delta = cfg.a_plus * math.exp(-dt / cfg.tau_plus)
            return STDPDelta(weight_delta=delta, pre_spike=True, post_spike=False, timing_diff=dt)
        elif dt < 0:
            # Depression: pre fires AFTER post → weaken
            delta = -cfg.a_minus * math.exp(dt / cfg.tau_minus)  # dt is negative
            return STDPDelta(weight_delta=delta, pre_spike=False, post_spike=True, timing_diff=dt)
        else:
            return STDPDelta(weight_delta=0.0, pre_spike=False, post_spike=False, timing_diff=0.0)

    def apply_to_weights(self, weights: list[list[float]], pre_spikes: list[float], post_spikes: list[float]) -> list[list[float]]:
        """Apply STDP updates to a weight matrix given spike times.

        Args:
            weights: current weight matrix [out_neurons x in_neurons]
            pre_spikes: spike times for presynaptic neurons
            post_spikes: spike times for postsynaptic neurons

        Returns:
            updated weight matrix
        """
        cfg = self.config
        new_weights = [row[:] for row in weights]

        for i in range(len(weights)):       # postsynaptic (output) neuron
            for j in range(len(weights[i])):  # presynaptic (input) neuron
                for t_post in post_spikes[i]:
                    for t_pre in pre_spikes[j]:
                        delta = self.compute_delta(t_pre, t_post)
                        new_weights[i][j] = min(
                            cfg.w_max, max(cfg.w_min, new_weights[i][j] + delta.weight_delta)
                        )

        return new_weights


class PoissonEncoder:
    """Encode continuous values as Poisson spike trains."""

    def __init__(self, rate: float = 100.0, seed: int | None = None):
        """Initialize encoder.

        Args:
            rate: firing rate in Hz
            seed: optional random seed for reproducibility
        """
        self.rate = rate
        self._rng = seed if seed is not None else None

    def encode(self, values: list[float], timesteps: int, dt: float = 1.0) -> list[list[bool]]:
        """Encode floating point values as spike trains.

        Args:
            values: input values to encode (normalized 0..1 typically)
            timesteps: number of simulation timesteps
            dt: timestep in ms

        Returns:
            spike_train: list of lists of bool [timesteps x n_neurons]
        """
        rng = self._rng
        import random
        if rng is not None:
            random.seed(rng)

        spike_train = []
        for t in range(timesteps):
            step_spikes = []
            for v in values:
                # Probability of spike per timestep = rate * dt / 1000
                prob = min(1.0, self.rate * dt / 1000.0)
                spiked = random.random() < prob * max(0.0, min(1.0, v))
                step_spikes.append(spiked)
            spike_train.append(step_spikes)

        return spike_train


class RateDecoder:
    """Decode spike trains back to continuous rates."""

    def __init__(self, window_size: int | None = None):
        self.window_size = window_size

    def decode(self, spike_train: list[list[bool]], timesteps: int | None = None) -> list[float]:
        """Convert spike train to firing rates.

        Args:
            spike_train: list of lists of bool [timesteps x n_neurons]
            timesteps: optional override for number of timesteps

        Returns:
            list of firing rates per neuron
        """
        if not spike_train:
            return []

        n_neurons = len(spike_train[0])
        n_steps = timesteps if timesteps is not None else len(spike_train)

        if self.window_size is not None and self.window_size < n_steps:
            # Use only the last window_size timesteps
            start_idx = max(0, n_steps - self.window_size)
            window = spike_train[start_idx:n_steps]
        else:
            window = spike_train

        rates = []
        for neuron_idx in range(n_neurons):
            spike_count = sum(1 for step in window if step[neuron_idx])
            rates.append(spike_count / len(window) if window else 0.0)

        return rates


class SpikeTrain:
    """A spike train — binary spike pattern over time."""

    def __init__(self, data: list[list[bool]] | None = None, neurons: int = 0, timesteps: int = 0):
        if data:
            self.data = data
            self.neurons = len(data[0]) if data else 0
            self.timesteps = len(data)
        else:
            self.neurons = neurons
            self.timesteps = timesteps
            self.data = [[False] * neurons for _ in range(timesteps)]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.timesteps, self.neurons)

    def to_spike_counts(self) -> list[float]:
        """Sum spikes per neuron across all timesteps."""
        counts = [0.0] * self.neurons
        for step in self.data:
            for i, spiked in enumerate(step):
                if spiked:
                    counts[i] += 1.0
        return counts

    def to_rate_vector(self) -> list[float]:
        """Convert to firing rate vector."""
        counts = self.to_spike_counts()
        return [c / self.timesteps for c in counts]

    def __repr__(self) -> str:
        return f"SpikeTrain(shape={self.shape}, total_spikes={sum(self.to_spike_counts())})"


def lif_simulation(inputs: list[list[float]], n_neurons: int, config: LIFConfig | None = None) -> SpikeTrain:
    """Run a complete LIF simulation.

    Args:
        inputs: input current per timestep [timesteps x input_dim]
        n_neurons: number of neurons in the layer
        config: LIF configuration

    Returns:
        SpikeTrain with the output spike pattern
    """
    cfg = config or LIFConfig()
    layer = LIFLayer(n_neurons, cfg)

    input_dim = len(inputs[0]) if inputs else n_neurons
    layer.initialize_weights(input_dim)

    timesteps = len(inputs)
    spike_train = layer.forward_timesteps(inputs, timesteps)

    return SpikeTrain(data=spike_train, neurons=n_neurons, timesteps=timesteps)