# ChimeraLang CIR + Belief System — Design Spec
**Date:** 2026-04-27
**Status:** Approved

---

## 1. Overview

Extend ChimeraLang with a Cognitive Intermediate Representation (CIR) layer and
`belief`/`inquire`/`resolve`/`guard`/`evolve` constructs. The existing `fn`/`gate`/
`goal`/`reason` VM is untouched — both grammars coexist. Programs using `belief`
constructs are routed through the CIR path; existing programs use the existing VM.

Cross-referenced from: MLIR, Dempster-Shafer theory, Active Inference (Friston),
Probabilistic Programming (Pyro/Stan), Byzantine Fault Tolerance, Weisfeiler-Lehman
graph hashing, AGM belief revision, CRDT distributed systems.

---

## 2. New Files

```
chimera/cir/__init__.py
chimera/cir/nodes.py       — CIR graph: SSA-form nodes, BetaDist beliefs, typed edges
chimera/cir/lower.py       — AST → CIR lowering (3 passes)
chimera/cir/executor.py    — CIR executor: DS combination, BFT consensus, FE minimization
chimera/cir/symbols.py     — Symbol emergence: WL hashing, TF-IDF, Darwinian competition
examples/belief_reasoning.chimera
tests/test_cir_nodes.py
tests/test_cir_lower.py
tests/test_cir_executor.py
tests/test_symbols.py
```

**Modified files:** `chimera/lexer.py`, `chimera/parser.py`, `chimera/ast_nodes.py`,
`chimera/cli.py`

---

## 3. Grammar

### 3.1 New Tokens
`belief`, `inquire`, `resolve`, `guard`, `evolve`, `symbol`

### 3.2 New AST Nodes

```
BeliefDecl(name, type_ann, inquire_expr)
InquireExpr(prompt, agents, ttl)
ResolveStmt(target, threshold, strategy)   # strategy: dempster_shafer | majority | bft
GuardStmt(target, max_risk, strategy)      # strategy: variance | mean | both
EvolveStmt(target, condition, max_iter)
SymbolDecl(name, body)
```

### 3.3 Example Syntax

```chimera
belief cause := inquire {
  prompt: "What causes black holes?",
  agents: [claude],
  ttl: 3600
}

resolve cause with consensus { threshold: 0.8, strategy: dempster_shafer }
guard cause against hallucination { max_risk: 0.2, strategy: both }
evolve cause until stable { max_iter: 3 }

emit cause
```

---

## 4. CIR Node Types (`chimera/cir/nodes.py`)

### 4.1 BeliefState

```python
@dataclass
class BetaDist:
    alpha: float   # pseudocounts for success
    beta: float    # pseudocounts for failure

    @property
    def mean(self) -> float: return self.alpha / (self.alpha + self.beta)
    @property
    def variance(self) -> float:
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s * s * (s + 1))

    def combine_ds(self, other: BetaDist) -> BetaDist:
        """Dempster-Shafer combination (not weighted average)."""
        ...

    def kl_divergence(self, other: BetaDist) -> float:
        """KL(self || other) for free energy minimization."""
        ...

@dataclass
class BeliefState:
    id: UUID
    name: str
    distribution: BetaDist
    ttl: float | None
    provenance: list[str]
    timestamp: float
    edges_out: list[CIREdge]
```

### 4.2 CIR Node Types

| Node | Role |
|---|---|
| `InquiryNode` | LLM call via claude_adapter → BeliefState |
| `ConsensusNode` | DS-combines N BeliefStates |
| `ValidationNode` | Guard: checks mean ≥ (1-max_risk) AND variance ≤ 0.05 |
| `EvolutionNode` | Fixed-point loop: minimize KL divergence |
| `MetaNode` | Graph introspection, emits ReasoningTrace |

### 4.3 Edge Types

`INQUIRY | CONSENSUS | VALIDATION | EVOLUTION | CAUSAL`

Each edge carries `strength: float` for causal weight.

---

## 5. AST → CIR Lowering (`chimera/cir/lower.py`)

Three sequential passes:

**Pass 1 — Structural:** BeliefDecl→InquiryNode, ResolveStmt→ConsensusNode,
GuardStmt→ValidationNode, EvolveStmt→EvolutionNode. Nodes get UUIDs. Edges typed.

**Pass 2 — Dead belief elimination:** Remove nodes with no emit/resolve downstream.
Mirrors LLVM DCE on belief flow graph.

**Pass 3 — Belief flow analysis:** Forward-propagate BetaDist parameters. Flag nodes
where variance exceeds max_risk before reaching ValidationNode.

---

## 6. CIR Executor (`chimera/cir/executor.py`)

### 6.1 Inquiry
Calls `ClaudeAdapter.query(prompt)`. Maps confidence → `BetaDist(alpha=conf*10, beta=(1-conf)*10)`.

### 6.2 Resolve — Dempster-Shafer + BFT
DS combination rule across N belief masses. Conflict detection: if `K > 0.8` raise
`ConflictException`. BFT: if ≥ ⌊(N-1)/3⌋ agents produce wildly divergent beliefs,
invalidate round and retry (max 3 retries).

### 6.3 Guard — Strongest Postcondition
Pass condition: `mean ≥ (1 - max_risk)` AND `variance ≤ 0.05`. Fail → `GuardViolation`
with flagged belief and full provenance trace.

### 6.4 Evolve — Free Energy Minimization
Fixed-point loop: run subgraph → compute KL(posterior || prior) → rewrite CIR edges
to reduce KL. Terminate when `KL < 0.001` or `max_iter` reached. This is Free Energy
minimization per Friston's Active Inference framework.

### 6.5 Temporal Decay
At execution, beliefs older than TTL have their BetaDist regressed toward uniform
`(1, 1)` proportional to age overshoot. Stale ≠ wrong — just uncertain.

### 6.6 Metacognitive Layer
`MetaNode` emits: node count, cycle detection, belief entropy, bottleneck nodes
(high in-degree), and full `ReasoningTrace` as a first-class output artifact.

---

## 7. Symbol Emergence (`chimera/cir/symbols.py`)

### 7.1 Extraction
Enumerate all subgraphs (chains + branches) of length ≥ 2 after each execution.

### 7.2 Identity — Weisfeiler-Lehman Hashing
3-round WL hashing on (node_type, edge_type) neighborhoods. Same hash = structurally
equivalent = same symbol candidate. O(n·k) where k=3 rounds.

### 7.3 Semantic Similarity — TF-IDF
InquiryNode prompts tokenized and scored. Cosine similarity > 0.7 → subgraphs
merged into one symbol. Zero heavy dependencies (no torch/transformers).

### 7.4 Multi-Objective Fitness

```
fitness = 0.35 * compression_ratio
        + 0.25 * structural_depth
        + 0.20 * semantic_coherence
        + 0.20 * usage_frequency
```

### 7.5 Darwinian Competition
Every 10 uses: rank all symbols by fitness. Bottom 20% pruned. Survivors mutate
(edge weights ±0.05). New candidates from current run enter the pool.

### 7.6 CRDT Symbol Store
Grow-only G-Set CRDT keyed by WL hash. Merge = union (conflict-free).
`save_symbols(path: str)` → JSON. `load_symbols(path: str)` → merge into store.
CLI flags: `--save-symbols`, `--load-symbols`.

---

## 8. Integration

### 8.1 CLI Dispatch
After parsing: if AST contains any `BeliefDecl` → CIR path. Else → existing VM.

### 8.2 claude_adapter.py
`InquiryNode` calls existing `ClaudeAdapter`. No changes to adapter needed.

### 8.3 Backward Compatibility
Zero breaking changes. All existing `.chimera` programs run identically.

---

## 9. Testing Strategy

| File | Coverage target |
|---|---|
| `test_cir_nodes.py` | BetaDist math, DS combination, KL divergence |
| `test_cir_lower.py` | All 3 lowering passes, dead belief elimination |
| `test_cir_executor.py` | Inquiry mock, DS resolve, guard violations, evolve convergence |
| `test_symbols.py` | WL hashing, TF-IDF similarity, fitness, CRDT merge |

Target: 80%+ coverage across all new files.

---

## 10. Constraints

- No new heavy dependencies (no torch, no scipy, no numpy required)
- BetaDist math implemented from scratch using `math` stdlib
- WL hashing uses stdlib `hashlib`
- TF-IDF uses stdlib `collections.Counter`
- All new code is type-annotated (Python 3.11+)
