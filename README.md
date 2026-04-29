# ChimeraLang

**A programming language designed for AI cognition** — probabilistic types, quantum consensus gates, directed hallucination, cryptographic integrity proofs, and a Cognitive Intermediate Representation (CIR) with self-evolving symbol emergence.

ChimeraLang treats uncertainty, confidence, and epistemic state as **first-class language primitives** rather than bolted-on libraries. Programs in ChimeraLang describe *how an AI should think*, not just what it should compute.

---

## Key Features

| Feature | Description |
|---|---|
| **CIR — Cognitive Intermediate Representation** | A graph-based IR where beliefs flow as Beta distributions through Inquiry → Consensus → Validation → Evolution nodes |
| **Belief System** | `belief`/`inquire`/`resolve`/`guard`/`evolve` — first-class epistemic constructs backed by Dempster-Shafer evidence combination |
| **Probabilistic Types** | `Confident<T>`, `Explore<T>`, `Converge<T>`, `Provisional<T>` — types that carry confidence scores |
| **Quantum Consensus Gates** | Multiple candidate values vote under Gaussian noise; the result is the *consensus* of an ensemble |
| **Symbol Emergence** | Reusable CIR subgraphs discovered automatically via Weisfeiler-Lehman hashing + TF-IDF similarity, evolved by Darwinian fitness competition |
| **Hallucination Detection** | Inline `detect` blocks + guard nodes with variance-aware Beta distribution checks |
| **Cryptographic Integrity** | Merkle-chain proofs and gate certificates ensure reasoning traces are tamper-evident |
| **Temporal Belief Decay** | Beliefs with TTL decay toward uniform prior as they age — staleness is uncertainty, not error |
| **Memory Modifiers** | `Ephemeral`, `Persistent`, `Provisional` — explicit lifecycle for every binding |
| **Interactive REPL** | `chimera repl` — try the language live in your terminal |

---

## Quick Start

```bash
git clone https://github.com/fernandogarzaaa/ChimeraLang
cd ChimeraLang
python -m chimera.cli run examples/belief_reasoning.chimera --trace
```

**Requirements:** Python ≥ 3.11, no external dependencies. Install `anthropic` for live LLM `inquire` calls; otherwise runs with a mock adapter.

---

## Execution Paths

ChimeraLang has three backward-compatible execution paths:

| Path | Triggered by | Constructs |
|---|---|---|
| **CIR path** | Any `belief` declaration | `belief`, `inquire`, `resolve`, `guard`, `evolve`, `symbol` |
| **VM path** | All other programs | `fn`, `gate`, `goal`, `reason`, `val`, `for`, `match` |
| **Compiler path** | `chimera compile` | `model`, `layer`, `train`, `constitution`, `retrieval`, `MoE`, roadmap declarations |

Existing programs run identically. New CIR programs are automatically routed.

---

## The CIR Belief System

### Syntax

```chimera
belief cause := inquire {
  prompt: "What are the primary causes of black hole formation?",
  agents: [claude],
  ttl: 3600
}

resolve cause with consensus { threshold: 0.8, strategy: dempster_shafer }
guard cause against hallucination { max_risk: 0.2, strategy: both }
evolve cause until stable { max_iter: 3 }

emit cause
```

### Run it

```bash
python -m chimera.cli run examples/belief_reasoning.chimera --trace
```

```
  emit: cause  [mean=0.750 variance=0.0170]

— CIR Reasoning Trace —
  [inquiry] prompt='What are the primary causes...' agents=['claude']
  [inquiry] confidence=0.750 -> Beta(7.5,2.5)
  [consensus] strategy=dempster_shafer threshold=0.75
  [consensus] combined mean=0.750 variance=0.0170
  [guard] max_risk=0.25 strategy=both
  [guard] PASSED — mean=0.750 variance=0.0170
  [evolve] condition=stable max_iter=3

chimera: examples/belief_reasoning.chimera — CIR executed in 0.1ms
```

### Saving and reusing symbols

```bash
# First run: extract and save reusable subgraph symbols
python -m chimera.cli run program.chimera --save-symbols=symbols.json

# Later runs: load symbols to bootstrap belief patterns
python -m chimera.cli run program.chimera --load-symbols=symbols.json
```

---

## CIR Architecture

```
ChimeraLang Source
      │
   Lexer + Parser   (belief / inquire / resolve / guard / evolve / symbol)
      │
   AST              (BeliefDecl, InquireExpr, ResolveStmt, GuardStmt, EvolveStmt)
      │
   chimera/cir/
   ├── lower.py     — AST → CIR graph (3 passes: structural, dead belief elimination, flow analysis)
   ├── nodes.py     — BetaDist beliefs, InquiryNode, ConsensusNode, ValidationNode, EvolutionNode
   ├── executor.py  — DS combination, BFT guard, free energy evolve, temporal decay, Claude adapter
   └── symbols.py   — WL hashing, TF-IDF merge, multi-objective fitness, Darwinian competition, CRDT store
      │
   BeliefResult     (distribution + trace + guard violations + symbol log)
```

### How beliefs work

Beliefs are **Beta distributions** `Beta(α, β)` — not scalar floats. This means:

- `mean = α / (α + β)` — the estimated truth value
- `variance = αβ / ((α+β)²(α+β+1))` — how uncertain we are about the estimate
- Low pseudocounts = high variance = little evidence = uncertain belief
- `inquire` converts a confidence score to `Beta(conf×10, (1-conf)×10)`

### Dempster-Shafer consensus (`resolve`)

`resolve` combines N beliefs using DS evidence combination — not a naive weighted average. When two sources conflict (one says very high, other says very low), a `ConflictException` is raised rather than silently averaging to 0.5.

### Guard (`guard`)

`guard` checks: `mean ≥ (1 − max_risk)` AND `variance ≤ 0.05`. A belief that's above the mean threshold but wildly uncertain still fails the variance check.

### Free Energy evolution (`evolve`)

`evolve` runs a fixed-point loop minimizing KL divergence between successive belief updates — inspired by Friston's Active Inference framework. Terminates when `KL < 0.001` or `max_iter` reached.

### Symbol Emergence

After each execution, ChimeraLang automatically extracts reusable CIR subgraphs:

1. **Weisfeiler-Lehman hashing** — structural identity across different prompts
2. **TF-IDF cosine similarity** — semantically similar subgraphs (score > 0.7) are merged
3. **Multi-objective fitness** — `0.35×compression + 0.25×depth + 0.20×coherence + 0.20×usage`
4. **Darwinian competition** — every 10 uses, bottom 20% by fitness are pruned; survivors mutate
5. **CRDT G-Set store** — conflict-free distributed symbol library (merge = union)

---

## VM Path (Existing Language)

### Quantum Consensus Gates

```chimera
gate consensus_answer(question: Text) -> Converge<Text>
  branches: 5
  collapse: weighted_vote
  threshold: 0.80
  val answer: Text = "Reasoned answer to: " + question
  return answer
end

val result = consensus_answer("What causes consciousness?")
```

### Probabilistic Types

```chimera
val answer: Confident<Int> = confident(42, 0.95)
val idea:   Explore<Text>  = explore("maybe this?", 0.60)
```

### For Loops + Match

```chimera
val scores = [0.92, 0.76, 0.88]
for s in scores
  emit s
end

match status
  | 1 => emit "running"
  | _ => emit "unknown"
end
```

### Hallucination Detection

```chimera
detect hallucination
  strategy: "range"
  on: temperature
  valid_range: [-50.0, 60.0]
  action: "flag"
end
```

---

## Examples

| File | What it demonstrates |
|---|---|
| `belief_reasoning.chimera` | **Full CIR pipeline**: belief → inquire → resolve → guard → evolve → emit |
| `hello_chimera.chimera` | Basic emit, confident values |
| `quantum_reasoning.chimera` | Consensus gates, confidence propagation |
| `goal_driven.chimera` | Goals, reasoning blocks, semantic constraints |
| `hallucination_guard.chimera` | All 5 hallucination-detection strategies |
| `for_loop.chimera` | For loops, list builtins, match expressions |
| `advanced_reasoning.chimera` | Detect blocks, nested gates + reason |

---

## CLI Reference

```bash
python -m chimera.cli run    <file> [--trace] [--save-symbols=out.json] [--load-symbols=in.json]
python -m chimera.cli check  <file>          # Type-check without running
python -m chimera.cli prove  <file>          # Run + generate integrity proof
python -m chimera.cli compile <file> [--backend=pytorch|llvm] [--out=file]
python -m chimera.cli parse  <file>          # Print AST
python -m chimera.cli lex    <file>          # Print token stream
python -m chimera.cli repl                   # Interactive REPL
```

---

## Production Status

The ML roadmap surface in `docs/roadmap/CHIMERALANG-ML-SPEC-V2.md` is implemented as a production-ready alpha:

| Area | Status |
|---|---|
| Language surface | Parser and AST support for tensor metadata, vector stores, spike trains, multimodal types, memory pointers, retrieval blocks, causal models, federated training, meta-learning, self-improvement, swarms, replay buffers, rewards, and predictive coding |
| Validation | Type checker rejects invalid dimensions, retrieval settings, roadmap declarations, and constitution schemas before generation |
| PyTorch backend | Generates executable modules for dense networks, MoE routing, retrieval stores, and roadmap-aware model metadata |
| LLVM backend | `chimera compile --backend=llvm` emits typed LLVM IR skeletons for model declarations |
| Runtime package | `chimera_runtime` exports vector storage, spiking runtime primitives, swarm coordination, and roadmap system containers |
| CI and packaging | GitHub Actions run tests on Python 3.11-3.13 and build wheel/sdist artifacts |

Roadmap details and verification notes live in `docs/roadmap/IMPLEMENTATION-STATUS.md`.

---

## Project Structure

```
ChimeraLang/
├── chimera/
│   ├── cir/                  # Cognitive Intermediate Representation
│   │   ├── nodes.py          # BetaDist, CIR node types, CIRGraph + WL hash
│   │   ├── lower.py          # AST → CIR lowering (3 passes)
│   │   ├── executor.py       # DS combination, BFT guard, evolve, temporal decay
│   │   ├── symbols.py        # Symbol emergence + CRDT store
│   │   └── __init__.py       # run_cir() public API
│   ├── tokens.py             # 85+ token types incl. belief/inquire/resolve/guard/evolve
│   ├── lexer.py              # Tokenizer (incl. := walrus operator)
│   ├── ast_nodes.py          # AST node hierarchy incl. BeliefDecl, InquireExpr
│   ├── parser.py             # Recursive-descent parser (both paths)
│   ├── types.py              # Runtime type system & confidence propagation
│   ├── vm.py                 # Quantum Consensus VM (fn/gate/goal/reason path)
│   ├── detect.py             # Hallucination detector
│   ├── integrity.py          # Merkle chains & gate certificates
│   └── cli.py                # CLI + REPL + automatic CIR/VM dispatch
├── examples/
├── spec/SPEC.md
├── paper/chimeralang.tex
├── tests/                    # 106 tests (60 VM + 46 CIR)
└── pyproject.toml
```

---

## How It Differs

| Aspect | Traditional Languages | ChimeraLang |
|---|---|---|
| Values | Deterministic | Beta distributions carrying full uncertainty |
| Evidence combination | N/A | Dempster-Shafer (conflict-aware, not naive averaging) |
| Execution | Single-path | Ensemble consensus + CIR belief graph |
| Correctness | Tests/assertions | Continuous guard nodes + hallucination detection |
| Auditability | Logs | Cryptographic Merkle proofs + full reasoning trace |
| Learning | None | Symbol emergence — reusable patterns emerge from execution history |
| Staleness | N/A | Temporal decay — old beliefs become uncertain, not wrong |

---

## License

MIT

## Citation

```bibtex
@article{chimeralang2025,
  title   = {ChimeraLang: A Programming Language for AI Cognition},
  year    = {2025},
  note    = {https://github.com/fernandogarzaaa/ChimeraLang}
}
```
