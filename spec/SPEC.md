# ChimeraLang Specification v0.2.0

## Abstract

ChimeraLang is a programming language designed exclusively for AI cognition. Unlike
traditional languages built for human sequential reasoning, ChimeraLang encodes
probabilistic reasoning, semantic constraints, quantum consensus, and memory scoping
as first-class language primitives.

The language does not eliminate hallucination — it *harnesses* it. Every computation
is a controlled exploration of possibility space, collapsed into committed output
through quantum-inspired consensus gates.

---

## 1. Design Principles

1. **Directed Hallucination** — Creativity is not a bug. Exploration is a primitive.
   The language controls *where* and *how much* an AI may explore, not *whether*.
2. **Probabilistic Types** — Every value carries confidence metadata. The type system
   enforces certainty boundaries at compile time.
3. **Semantic Constraints** — Functions declare *what the output must mean*, not
   *how to compute it*. The runtime searches for satisfying outputs.
4. **Quantum Consensus** — Multi-path reasoning with collapse. N branches explore,
   a consensus gate commits one answer.
5. **Cognitive Transparency** — Every execution produces a reasoning trace as a
   first-class artifact, enabling hallucination fingerprinting and visual inspection.

---

## 2. Type System

### 2.1 Primitive Types

```
Int, Float, Bool, Text, Void
```

### 2.2 Probabilistic Type Constructors

| Constructor       | Semantics                                          | Confidence Requirement |
|-------------------|----------------------------------------------------|------------------------|
| `Confident<T>`   | Must resolve with high certainty                   | >= 0.95                |
| `Explore<T>`     | Allowed to hallucinate within bounds               | any                    |
| `Converge<T>`    | Multiple paths, must reach consensus               | >= threshold           |
| `Provisional<T>` | Held as true until contradicted                    | any, revocable         |

### 2.3 Memory Type Modifiers

| Modifier       | Semantics                              |
|----------------|----------------------------------------|
| `Ephemeral`    | Working memory, forgotten after scope  |
| `Persistent`   | Long-term, survives across executions  |
| `Provisional`  | Held until explicitly contradicted     |

### 2.4 Composite Types

```
List<T>, Map<K, V>, Option<T>, Result<T, E>
```

---

## 3. Syntax

### 3.1 Variable Declarations

```chimera
val name: Confident<Text> = "ChimeraLang"
val hypothesis: Explore<Text>               // AI fills this
val route: Converge<Plan>                    // consensus required
val temp: Ephemeral<Int> = 42
val belief: Persistent<KnowledgeGraph>
val assumption: Provisional<Fact>
```

### 3.2 Functions with Semantic Constraints

```chimera
fn summarize(doc: Text) -> Confident<Text>
  must: len(output) < len(input) * 0.3
  must: preserve(key_entities(input))
  allow: rephrase, infer, compress
  forbidden: invent_facts
end
```

### 3.3 Quantum Consensus Gates

```chimera
gate decide(question: Text) -> Confident<Text>
  branches: 5
  collapse: majority
  threshold: 0.85
  fallback: escalate
end
```

### 3.4 Goal Declarations (Intent-First Execution)

```chimera
goal "produce a valid API contract"
  constraints: [REST, idempotent, authenticated]
  quality: security > performance > readability
  explore_budget: 3.0
end
```

### 3.5 Reasoning Blocks

```chimera
reason about(problem: Text) -> Explore<Solution>
  given: [context, constraints]
  explore: divergent_paths(5)
  evaluate: score_by(quality_axes)
  commit: highest_consensus
end
```

### 3.6 Integrity Assertions

```chimera
assert confident(result) >= 0.95
assert consensus(gate_output) >= threshold
assert no_hallucination(output, source_facts)
```

---

## 4. Execution Model

### 4.1 Quantum Consensus Execution

Every `gate` spawns N reasoning branches in superposition. Each branch independently
computes a result. The collapse function (majority, weighted_vote, highest_confidence)
selects the committed output.

```
[Input] → spawn(N branches) → parallel_reason → collapse → [Committed Output]
```

### 4.2 Confidence Propagation

Confidence flows through the computation graph:
- Operations on `Confident<T>` values preserve high confidence
- Mixing `Confident` with `Explore` degrades to the lower confidence
- `Converge` requires consensus above threshold to produce `Confident` output
- `Provisional` values are automatically flagged in integrity reports

### 4.3 Memory Lifecycle

- `Ephemeral` values are garbage-collected at scope exit
- `Persistent` values are serialized to the knowledge store
- `Provisional` values are tracked and can be invalidated by contradiction

---

## 5. Hallucination Detection

The runtime tracks reasoning fingerprints:

1. **Divergence detection** — paths that deviate from consensus are flagged
2. **Source tracing** — every fact traces back to a source (input, derived, generated)
3. **Confidence anomaly** — high local confidence with low cross-path agreement
4. **Promotion violation** — `Confident` type derived from `Provisional` source

---

## 6. Integrity Proofs

Every execution produces an `IntegrityReport`:

```
INTEGRITY REPORT
━━━━━━━━━━━━━━━━
Paths explored:     N
Consensus reached:  M/N (percentage)
Hallucination risk: LOW | MEDIUM | HIGH | CRITICAL
Divergence points:  [list of steps]
Resolution:         [collapse method] → committed path verified
Proof hash:         SHA-256 of reasoning trace
```

---

## 7. File Extension

`.chimera`

---

## 8. Reserved Keywords

### 8.1 Core control flow & declarations

```
val, fn, gate, goal, reason, about, end, return,
if, else, match, for, in, true, false, and, or, not,
import, from, as
```

### 8.2 Semantic-constraint vocabulary

```
must, allow, forbidden, given, explore, evaluate, commit,
branches, collapse, threshold, fallback, constraints, quality,
explore_budget, majority, weighted_vote, highest_confidence,
escalate, with, against, until
```

### 8.3 CIR runtime hooks (see §11)

```
emit, detect, belief, inquire, resolve, guard, evolve, symbol, on
```

### 8.4 ML / model surface (see §12)

```
model, layer, train, forward, moe, constitution
```

### 8.5 Type constructors

```
Confident, Explore, Converge, Provisional, Ephemeral, Persistent,
Int, Float, Bool, Text, Void, List, Map, Option, Result,
Tensor, VectorStore, SpikeTrain, Multimodal, MemPtr, Grad, Constituted
```

### 8.6 Built-in functions

```
assert, confident, consensus, no_hallucination, preserve, len,
key_entities, divergent_paths, score_by, highest_consensus
```

---

## 9. Translator Architecture

```
┌─────────────────────────────────────────┐
│        AI LAYER (ChimeraLang)           │
│  probabilistic · quantum · semantic     │
└──────────────────┬──────────────────────┘
                   │
          [ Chimera Compiler ]
          semantic compression
          integrity proofs
          hallucination detection
                   │
┌──────────────────▼──────────────────────┐
│       HUMAN LAYER (ChimeraView)         │
│  compressed · observable · visual       │
│  reasoning graph · integrity report     │
└─────────────────────────────────────────┘
```

---

## 10. Bootstrapping

ChimeraLang v0.1 is bootstrapped in Python. The self-improvement loop begins
when the language can compile itself — at which point AI agents write ChimeraLang
v(n+1) using ChimeraLang v(n), with each iteration encoding discovered failure
modes as unrepresentable constructs.

---

## 11. CIR Runtime Hooks

The Computational Intelligence Representation (CIR) layer adds first-class
operators for memory, retrieval, evolution, and integrity tracking. Each
hook compiles down to a runtime call recorded in the reasoning trace.

### 11.1 `emit`

```chimera
emit value
```

Records `value` as a committed output of the current scope. Every `emit`
is captured by the integrity report (§6) with a confidence band and source.

### 11.2 `detect`

```chimera
detect hallucination(output, source_facts)
```

Runs the hallucination scanner against `output` using `source_facts` as
the trusted ground set. Returns a `Confident<Bool>` plus a populated
`violations` list when the scan trips.

### 11.3 `belief` / `inquire` / `resolve`

```chimera
belief b: Persistent<Fact> = "the sky is blue"
val q: Confident<Text> = inquire(b, "what color is the sky?")
resolve b with new_evidence
```

`belief` declares a mutable, traced fact; `inquire` queries an adapter
(LLM, vector store, or oracle) and binds the answer with confidence
metadata; `resolve` updates the belief, recording the prior and posterior
in the reasoning trace so that contradictions are auditable.

### 11.4 `guard`

```chimera
guard output against forbidden_topics until confident(output) >= 0.95
```

Wraps a scope in a runtime check. If any contained value violates
`forbidden_topics`, the scope re-executes; if `until` is unmet within
the surrounding `explore_budget`, the guard escalates per its enclosing
`gate.fallback`.

### 11.5 `evolve`

```chimera
evolve plan with feedback
  branches: 4
  collapse: highest_confidence
end
```

Applies a mutation/critique loop to `plan`, generating `branches`
candidate revisions, scoring them, and committing under `collapse`.
Each iteration is logged for self-improvement audits.

### 11.6 `symbol`

```chimera
symbol concept = "operational_pattern"
```

Declares a reusable, persistent symbol whose embedding is stored in the
symbol store and made available for cross-program reuse. Symbols carry
a Beta(α, β) prior over their utility and update on observation.

---

## 12. Models, Layers & Training

ChimeraLang treats neural models as declarative graphs with structural
type checks at compile time and PyTorch / LLVM emission at runtime.

### 12.1 Model declarations

```chimera
model Classifier {
  layer fc1: Dense(784 -> 128)
  layer act1: ReLU(128 -> 128)
  layer fc2: Dense(128 -> 10)
  layer probs: Softmax(10 -> 10)
  constitution: ClassifierEthics
  device: cpu
}
```

Layer chains are dimension-validated: each layer's `in_dim` must equal
the previous layer's `out_dim`, or compilation fails (§4.4 in
`docs/roadmap`). Optional fields:

| Field          | Purpose                                          |
|----------------|--------------------------------------------------|
| `forward`      | Override the default sequential forward function |
| `retrieval`    | Attach a `VectorStore` for RAG-style models      |
| `uncertainty`  | One of `none`, `epistemic`, `aleatoric`          |
| `constitution` | Reference a `constitution` block for guarded outputs |
| `device`       | Target device (`cpu`, `cuda`, …)                 |

### 12.2 Built-in layer kinds

`Dense`, `ReLU`, `Softmax`, `LayerNorm`, `MoE` (mixture-of-experts with
`n_experts` and `top_k` config). Unknown layer kinds compile to a
pass-through with an audit comment so prototypes stay runnable.

### 12.3 Training

```chimera
train Classifier on mnist {
  epochs: 10
  batch_size: 32
  optimizer: Adam(lr=0.001)
  loss: CrossEntropy
}
```

Emits a structured epoch/batch loop (PyTorch backend) or a stack-allocated
LLVM IR scaffold (LLVM backend; see §15.2 for status).

### 12.4 Constitutions

```chimera
constitution ClassifierEthics {
  principles: ["Be helpful", "Be harmless", "Be honest"]
  rounds: 3
  max_violation_score: 0.05
}
```

Declares a critique loop that applies up to `rounds` redaction sweeps
to model outputs. `rounds` must be positive; the runtime layer clamps
`rounds=0` to 1 so the loop always runs at least once. Outputs include
a `rounds_run` count and a `violations` audit trail.

---

## 13. Modules & Imports

```chimera
import shared_facts from kb
import ConstitutionLayer from chimera_runtime as Layer
```

`import` resolves names against the import path (project root + standard
library) at parse time. `as` rebinds the imported symbol within the
current module. Circular imports are rejected.

---

## 14. Roadmap Systems

`chimera_runtime.roadmap_systems` exposes runtime primitives that map
1:1 to language-surface declarations. Each primitive's status:

| Primitive                  | Status            | Notes                                              |
|----------------------------|-------------------|----------------------------------------------------|
| `DifferentialPrivacyEngine`| Implemented       | Gradient clipping + Gaussian noise + budget tracking |
| `ReplayBuffer`             | Implemented       | Prioritised by priority key                        |
| `RewardSystem`             | Implemented       | Prediction-error reward                            |
| `PredictiveCodingRuntime`  | Implemented       | Layer-wise residual propagation                    |
| `ConstitutionLayer`        | Implemented       | Multi-round redact-and-rescan critique             |
| `CausalModel`              | **Stub**          | Returns `is_stub=True`; no inference yet           |
| `MetaLearner`              | **Stub**          | Returns `is_stub=True`; no inner-loop adaptation   |
| `SelfImprover`             | **Partial**       | Forbidden-op gate is real; verifier is not invoked (`verified=False`) |

Stub primitives validate constructor invariants and structural
preconditions but do not perform their stated computation. Their
result records carry explicit stub markers so callers can branch on
honesty rather than trusting a placeholder payload.

---

## 15. Surface Status

This document is the *intended* surface. Implementations live in the
following locations and may lag behind:

### 15.1 Lexer / parser / type checker

* Lexer: `chimera/tokens.py`
* Parser: `chimera/parser.py`
* AST: `chimera/ast_nodes.py`
* Type checker: `chimera/type_checker.py`

If a keyword in §8 is rejected by the lexer, the lexer is the source of
truth — file an issue.

### 15.2 Backends

* PyTorch codegen (`chimera/compiler/codegen_pytorch.py`) — generates
  runnable PyTorch modules from `model` / `train` blocks.
* LLVM IR backend (`chimera/compiler/llvm_backend.py`) — emits
  structural IR scaffolding. Several layer bodies still reference SSA
  values whose definitions are pending; the IR is dimension-checked
  and brace-balanced (regression-tested) but is **not yet `llvm-as`-clean**
  end-to-end.

### 15.3 Out-of-spec runtime

`chimera/cir/`, `chimera_runtime/symbol_store.py`, and the `inquire`
adapter contain experimental surface that ships ahead of this spec.
Treat anything not listed in §11–14 as unstable.

---

*ChimeraLang — Directed hallucination as a language primitive.*
