"""Swarm intelligence protocol — 16-agent gossip ring with specialist routing."""
from __future__ import annotations

import math
import time
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Agent:
    """A single swarm agent."""
    id: int
    domain: str  # "reasoning" | "code" | "science" | "creative"
    active: bool = True
    expertise_score: float = 0.5
    last_heartbeat: float = 0.0
    gradient_buffer: list = field(default_factory=list)


@dataclass
class GossipMessage:
    """Message exchanged between agents in the gossip ring."""
    sender_id: int
    receiver_id: int
    gradient_top_k: list  # top 1% of gradients (sparsified)
    timestamp: float
    round: int
    is_byzantine: bool = False  # true if this message is poisoned


@dataclass
class DomainExpertise:
    """Tracks which agents excel at which domains."""
    reasoning: list[int] = field(default_factory=list)
    code: list[int] = field(default_factory=list)
    science: list[int] = field(default_factory=list)
    creative: list[int] = field(default_factory=list)


@dataclass
class SwarmConfig:
    """Configuration for the swarm."""
    n_agents: int = 16
    topology: str = "ring"      # "ring" | "fully_connected"
    gossip_interval: int = 100  # steps between gossip rounds
    top_k_fraction: float = 0.01  # share top 1% of gradients
    byzantine_tolerance: int = 2  # number of poisoned agents we can handle
    compression: str = "topk"    # "topk" | "quantization" | "none"


class SwarmProtocol:
    """16-agent gossip ring with specialist domain routing."""

    def __init__(self, config: SwarmConfig | None = None):
        self.config = config or SwarmConfig()
        self.agents: list[Agent] = []
        self.expertise = DomainExpertise()
        self.message_log: list[GossipMessage] = []
        self.round = 0
        self._initialize_agents()

    def _initialize_agents(self) -> None:
        """Set up 16 agents with domain specialization."""
        domains = ["reasoning"] * 4 + ["code"] * 4 + ["science"] * 4 + ["creative"] * 4
        for i in range(self.config.n_agents):
            agent = Agent(
                id=i,
                domain=domains[i],
                active=True,
                expertise_score=0.5 + random.random() * 0.5,
                last_heartbeat=time.time()
            )
            self.agents.append(agent)

        self.expertise.reasoning = list(range(0, 4))
        self.expertise.code = list(range(4, 8))
        self.expertise.science = list(range(8, 12))
        self.expertise.creative = list(range(12, 16))

    def route_query(self, query: str) -> int:
        """Route a query to the most appropriate specialist agent.

        Args:
            query: the user query string

        Returns:
            agent_id of the selected specialist
        """
        # Simple keyword-based routing (in production this would use a model)
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["proof", "reason", "logic", "prove", "theorem"]):
            candidates = self.expertise.reasoning
        elif any(kw in query_lower for kw in ["code", "function", "class", "program", "debug", "implement"]):
            candidates = self.expertise.code
        elif any(kw in query_lower for kw in ["science", "experiment", "data", "hypothesis", "physics"]):
            candidates = self.expertise.science
        elif any(kw in query_lower for kw in ["creative", "art", "story", "design", "imagine"]):
            candidates = self.expertise.creative
        else:
            # Default to reasoning for ambiguous queries
            candidates = self.expertise.reasoning

        # Pick the agent with highest expertise score among candidates
        best_agent = min(candidates, key=lambda a: self.agents[a].id)
        best_score = -1.0
        for aid in candidates:
            if self.agents[aid].active and self.agents[aid].expertise_score > best_score:
                best_score = self.agents[aid].expertise_score
                best_agent = aid

        return best_agent

    def get_ring_neighbors(self, agent_id: int) -> tuple[int, int]:
        """Get the two neighbors in the ring topology."""
        n = self.config.n_agents
        left = (agent_id - 1) % n
        right = (agent_id + 1) % n
        return left, right

    def _compress_gradients(self, gradients: list[float]) -> list[float]:
        """Compress gradients using top-k sparsification."""
        if self.config.compression != "topk":
            return gradients

        k = max(1, int(len(gradients) * self.config.top_k_fraction))
        # Get indices of top-k values by absolute value
        indexed = [(abs(g), i, g) for i, g in enumerate(gradients)]
        indexed.sort(reverse=True)
        topk_indices = {idx for _, _, idx in indexed[:k]}

        # Return only top-k, rest set to 0
        compressed = [gradients[i] if i in topk_indices else 0.0 for i in range(len(gradients))]
        return compressed

    def gossip_round(self) -> list[GossipMessage]:
        """Execute one round of gossip in the ring.

        Returns:
            list of messages sent this round
        """
        messages = []
        self.round += 1

        for agent in self.agents:
            if not agent.active:
                continue

            left_neighbor, right_neighbor = self.get_ring_neighbors(agent.id)

            # Share gradient with left neighbor
            gradient = agent.gradient_buffer[-1] if agent.gradient_buffer else [0.0] * 100
            compressed = self._compress_gradients(gradient)

            # Byzantine detection: check for obviously malformed gradients
            is_byzantine = False
            if any(abs(g) > 100.0 for g in compressed):
                is_byzantine = random.random() < 0.05  # 5% chance of corruption for testing

            msg = GossipMessage(
                sender_id=agent.id,
                receiver_id=left_neighbor,
                gradient_top_k=compressed,
                timestamp=time.time(),
                round=self.round,
                is_byzantine=is_byzantine
            )
            messages.append(msg)

            # Also send to right neighbor
            msg2 = GossipMessage(
                sender_id=agent.id,
                receiver_id=right_neighbor,
                gradient_top_k=compressed,
                timestamp=time.time(),
                round=self.round,
                is_byzantine=is_byzantine
            )
            messages.append(msg2)

        self.message_log.extend(messages)
        return messages

    def aggregate_responses(self, agent_ids: list[int], responses: list[str]) -> str:
        """Aggregate responses from multiple specialists using quantum-style voting.

        Args:
            agent_ids: which agents responded
            responses: the actual response strings

        Returns:
            aggregated consensus response
        """
        if not responses:
            return ""

        if len(responses) == 1:
            return responses[0]

        # Score each response by agent expertise
        scored = []
        for aid, resp in zip(agent_ids, responses):
            agent = self.agents[aid] if aid < len(self.agents) else None
            score = agent.expertise_score if agent else 0.5
            scored.append((score, resp))

        # Sort by score descending and pick the best
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def detect_byzantine(self, messages: list[GossipMessage]) -> list[int]:
        """Detect and return IDs of agents sending poisoned messages.

        Args:
            messages: messages from the gossip round

        Returns:
            list of agent IDs identified as byzantine
        """
        byzantine_ids = []

        for msg in messages:
            # Simple detection heuristics
            if msg.is_byzantine:
                byzantine_ids.append(msg.sender_id)
                continue

            # Check for statistical anomalies in gradient values
            grads = msg.gradient_top_k
            if len(grads) > 0:
                mean_val = sum(grads) / len(grads)
                variance = sum((g - mean_val) ** 2 for g in grads) / len(grads)
                # If variance is extremely high, might be poisoned
                if variance > 1000.0:
                    byzantine_ids.append(msg.sender_id)

        return list(set(byzantine_ids))

    def quarantine_byzantine(self, byzantine_ids: list[int]) -> None:
        """Remove byzantine agents from active participation."""
        for bid in byzantine_ids:
            for agent in self.agents:
                if agent.id == bid:
                    agent.active = False
                    break

    def update_expertise_scores(self, domain: str, success: float) -> None:
        """Update expertise scores based on task success.

        Args:
            domain: which domain had success
            success: 0.0 to 1.0 score
        """
        expertise_map = {
            "reasoning": self.expertise.reasoning,
            "code": self.expertise.code,
            "science": self.expertise.science,
            "creative": self.expertise.creative,
        }

        agents = expertise_map.get(domain, [])
        for aid in agents:
            if aid < len(self.agents) and self.agents[aid].active:
                # Exponential moving average update
                old = self.agents[aid].expertise_score
                self.agents[aid].expertise_score = 0.9 * old + 0.1 * success

    def get_active_count(self) -> int:
        """Return the number of currently active (non-byzantine) agents."""
        return sum(1 for a in self.agents if a.active)

    def get_domain_agents(self, domain: str) -> list[int]:
        """Return all active agent IDs for a given domain."""
        expertise_map = {
            "reasoning": self.expertise.reasoning,
            "code": self.expertise.code,
            "science": self.expertise.science,
            "creative": self.expertise.creative,
        }
        candidates = expertise_map.get(domain, [])
        return [aid for aid in candidates if aid < len(self.agents) and self.agents[aid].active]

    def summary(self) -> dict:
        """Return swarm state summary."""
        domain_counts = {d: len(self.get_domain_agents(d)) for d in ["reasoning", "code", "science", "creative"]}
        return {
            "round": self.round,
            "total_agents": len(self.agents),
            "active_agents": self.get_active_count(),
            "byzantine_quarantined": len(self.agents) - self.get_active_count(),
            "domain_counts": domain_counts,
            "messages_exchanged": len(self.message_log),
        }