# ChimeraLang

**A programming language designed for AI cognition** — probabilistic types, ensemble consensus gates, hallucination detection, cryptographic integrity proofs, and a Cognitive Intermediate Representation (CIR) with self-evolving symbol emergence.

ChimeraLang treats uncertainty, confidence, and epistemic state as **first-class language primitives** rather than bolted-on libraries. Programs in ChimeraLang describe *how an AI should think*, not just what it should compute.

---

## Key Features

| Feature | Description |
|---|---|
| **CIR — Cognitive Intermediate Representation** | A graph-based IR where beliefs flow as Beta distributions through Inquiry → Consensus → Validation → Evolution nodes |
| **Belief System** | `belief`/`inquire`/`resolve`/`guard`/`evolve` — first-class epistemic constructs; `resolve` defaults to answer-agreement voting, with pseudocount-addition evidence combination (`pooled`) and opt-in logistic calibration |
| **Probabilistic Types** | `Confident<T>`, `Explore<T>`, `Converge<T>`, `Provisional<T>` — types that carry confidence scores |
| **Ensemble Consensus Gates** | N branches run with Gaussian-perturbed confidences and collapse by confidence-weighted vote |
| **Symbol Emergence** | Reusable CIR subgraphs discovered automatically via Weisfeiler-Lehman hashing + TF-IDF similarity, evolved by Darwinian fitness competition |
| **Hallucination Detection** | Inline `detect` blocks + guard nodes with variance-aware Beta distribution checks |
| **Cryptographic Integrity** | Merkle-chain proofs and gate certificates ensure reasoning traces are tamper-evident |
| **Temporal Belief Decay** | Beliefs with TTL decay toward uniform prior as they age — staleness is uncertainty, not error |
| **Memory Modifiers** | `Ephemeral`, `Persistent`, `Provisional` — explicit lifecycle for every binding |
| **Interactive REPL** | `chimera repl` — try the language live in your terminal |

---

## Installation

```bash
pip install chimeralang            # core (standard library only)
pip install "chimeralang[sign]"    # + Ed25519 certificate signing (cryptography)
```

This installs the `chimera` command. Ed25519 signing is optional — without the
`[sign]` extra it degrades gracefully, and verifying unsigned/HMAC certificates
still works.

```bash
chimera run    examples/belief_reasoning.chimera --trace   # execute
chimera check  examples/quantum_reasoning.chimera          # type + capability check
chimera prove  examples/quantum_reasoning.chimera --out=cert.json   # emit certificate
chimera verify cert.json                                   # verify offline
```

**Requirements:** Python ≥ 3.11. The core install has no third-party dependencies.
Install `anthropic` for live LLM `inquire` calls (otherwise a mock adapter is used).

## Quick Start (from source)

```bash
git clone https://github.com/fernandogarzaaa/ChimeraLang
cd ChimeraLang
python -m chimera.cli run examples/belief_reasoning.chimera --trace
```

---

## Execution Paths

ChimeraLang has three backward-compatible execution paths:

| Path | Triggered by | Constructs |
|---|---|---|
| **CIR path** | Any `belief` declaration | `belief`, `inquire`, `resolve`, `guard`, `evolve`, `symbol` |
| **VM path** | All other programs | `fn`, `gate`, `goal`, `reason`, `val`, `for`, `match` |
| **Compiler path** | `chimera compile` | `model`, `layer`, `train`, `constitution`, `retrieval`, `MoE`, roadmap declarations |
| **RAG path** | `chimera rag` | JSON corpus retrieval, cited extractive answers, confidence guards, constitution checks |

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

resolve cause with consensus { threshold: 0.75 }
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
  [inquiry] answer recorded (72 chars)
  [consensus] strategy=agreement threshold=0.75
  [consensus] single source, no combination performed
  [guard] max_risk=0.25 strategy=both
  [guard] PASSED — mean=0.750 variance=0.0170
  [evolve] condition=stable max_iter=3
  [evolve] iter=1 KL=0.00000 mean=0.750

— Lowering Warnings —
  guard on 'cause' has an unreachable variance limit (0.05); inquiry-produced beliefs cap variance at 0.0227 (strength 10), so the variance check can never fire

chimera: examples/belief_reasoning.chimera — CIR executed in 0.1ms
```

### Guard dominance

Every effectful CIR node (`evolve`, `emit`) should be dominated by a
validation node on its belief lineage: every belief-flow path from a
source to the node must pass through a `guard` positioned after any
`resolve`. A guard before a resolve does not dominate the consensus
(intentional conservatism). By default an undominated effectful node
produces a lowering warning; `--require-dominance` turns it into a
`LoweringError`. Under `--require-dominance`, a `mean`/`both` guard
without a calibrator is also an error (numeric thresholds on
uncalibrated posterior means are unsound); use `strategy: variance`
or supply `--calibrator=<path>`.

```bash
python -m chimera.cli run examples/guarded_pipeline.chimera --require-dominance
python -m chimera.cli run program.chimera --require-dominance --strict-guard --cert-out=cert.json
python -m chimera.cli verify cert.json
```

`--cert-out` writes a `chimeralang-cert/v2` certificate with a `cir`
section: the program source, the lowered graph, per-guard
`score_source` (`calibrated`/`uncalibrated`), the `strict_guard` run
flag (recorded for information only), and a dominance claim
(`enforced` only when dominance holds and every guard on every
dominating path is source-level `strict: true`; otherwise
`non-blocking` or `absent`).
`chimera verify` recomputes dominance from the embedded graph and
never trusts the stored claim or the `strict_guard` flag. A
source-level strict guard (`guard x against hallucination {
max_risk: 0.2, strict: true }`) makes a failing guard raise
`GuardViolation` and halt the run, like `--strict-guard`, but it is
part of the program source and therefore determines the certificate
claim. See `docs/design/guard-dominance.md` and
`examples/strict_guarded_pipeline.chimera`.

**What "enforced" means.** The `enforced` claim is structural only: it
attests that every effectful node is dominated by source-level strict
guards. It does NOT guarantee the guards are meaningful or calibrated.
A guard with `max_risk: 1.0` can never fail (it requires score >= 0)
yet still yields `enforced`. The certificate's `dominance.guards`
list flags each dominating guard as `vacuous` (can never fail) and
records its `score_source`; `dominance.guard_strength` is `vacuous`,
`uncalibrated`, or `nonvacuous` (worst case). `chimera verify` prints
a WARNING for vacuous or uncalibrated guards. Uncalibrated guards use
raw posterior means; the H1 confirmatory experiment found agreement
AUROC 0.6714 (CI [0.6280, 0.7185]) in Mode A and 0.7519 (CI [0.7121,
0.7908]) in Mode B, well below perfect discrimination. Lowering warns
on vacuous guards; `--require-dominance` makes a vacuous dominating
guard a `LoweringError`.

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

### Evidence combination (`resolve`)

`resolve` accepts three strategies. `agreement` (the default when a
belief has more than one source with answers) is vote-share over
normalized answers: the winner is the most common normalized answer,
ties broken by earliest source in agent order. The resolved belief
carries raw agreement (votes/N), a Laplace-smoothed Beta posterior
(unanimous 3/3 gives Beta(4,1), mean 0.8, never 1.0), and the winning
answer text. `pooled` chains `BetaDist.combine_pseudocount`: pure
pseudocount addition (`alpha + alpha'`, `beta + beta'`) after a
Dempster-style conflict check; when the conflict coefficient K exceeds
the threshold, a conflict error is raised rather than silently merging.
`dempster_shafer` is accepted as an alias of `pooled` with a lowering
warning. This was never formal Dempster-Shafer combination, and
`BetaDist.combine_ds` remains as a backward-compatible alias for
`combine_pseudocount`.

Why agreement is the default: the H1 experiment
(`experiments/h1_pooling`) found verbalized confidence near chance
(mode-B AUROC 0.51 to 0.54) while answer agreement carried the signal,
so multi-source beliefs with answers resolve by vote share unless
`strategy: pooled` is given explicitly. Single-source resolve passes
through unchanged whatever the strategy.

Confirmatory check (`PREREG_V2_CONFIRM`, SimpleQA questions 501-1000,
4,000 Nebius calls, 2026-10-03): agreement replicated as an informative
signal on fresh questions (mode A AUROC 0.6714, mode B AUROC 0.7519).
Calibration is opt-in and was shown to help for five-sample single-model
resolution (95% CI of constant-minus-calibrated Brier [0.0156, 0.0468])
but not shown for cross-model resolution (95% CI [-0.0007, 0.0300]).
Verdict: Inconclusive under the frozen rules; see
`experiments/h1_pooling/REPORT_V2_CONFIRM.md`. These numbers are scoped
to SimpleQA and the three models used
(`Qwen/Qwen3-235B-A22B-Instruct-2507`, `deepseek-ai/DeepSeek-V4-Pro`,
`google/gemma-3-27b-it`); they do not establish generalization beyond
them.

**Opt-in calibration.** A resolved belief's score is uncalibrated by
default (lowering warns loudly). `chimera/cir/calibration.py` provides
`LogisticCalibrator`, a deterministic pure-Python logistic fit of
calibrated probability against the uncalibrated score; pass it as
`run_cir(..., calibrator=...)` or `--calibrator=cal.json` on the CLI
and resolved beliefs carry `calibrated_p`, which guard and emit use in
place of the posterior mean. Calibration is only meaningful with a
held-out calibration set: fitting and evaluating on the same questions
gives an optimistic calibrator, and `fit` refuses fewer than
`MIN_FIT_N = 30` points.

**Multi-agent fan-out.** A belief with `agents: [a, b]` lowers to one `InquiryNode` per agent, so `resolve` now causes N adapter calls (one per agent) instead of one. Each agent's belief is tracked separately as `x@a`, `x@b` and pooled by the chain above. Different agents are *less correlated*, not independent: models share training data, so pooling still overstates the evidence somewhat, just less than repeated calls to one model. The default Anthropic adapter maps agent names to model ids through an explicit dict (`CIRExecutor(agent_models={...})`, default `{"claude": "claude-sonnet-4-6"}`) and raises a clear error naming any unknown agent rather than silently falling back to one model.

Without `resolve`, there is no pooled belief: `guard`, `evolve`, and `emit` apply to the *first* agent's belief only, and lowering warns loudly telling you to add `resolve`. Listing the same agent twice warns that the sources are the same model and the evidence is correlated.

### Guard (`guard`)

`guard` checks: `mean >= (1 - max_risk)` AND `variance <= max_variance`. `max_variance` is optional and defaults to 0.05 for backward compatibility. Note the default 0.05 can never fire on inquiry-produced beliefs: `BetaDist.from_confidence` uses strength 10, which caps variance at about 0.0227. The lowering pass warns when a guard's variance limit is unreachable. Set an explicit `max_variance` (e.g. `guard v against hallucination { max_risk: 0.2, strategy: variance, max_variance: 0.01 }`) for an enforceable variance check.

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

### Ensemble Consensus Gates

A `gate` runs N branches with Gaussian-perturbed input confidences and
collapses them by confidence-weighted vote. "Quantum" appears nowhere in
the syntax or semantics; the mechanism is a stochastic ensemble gate,
not a quantum algorithm.

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
python -m chimera.cli prove  <file> [--out=cert.json] [--key=hmac.key] [--sign-key=ed25519.pem]
python -m chimera.cli verify <cert.json> [--key=hmac.key] [--pubkey=HEX]
python -m chimera.cli compile <file> [--backend=pytorch|llvm] [--out=file]
python -m chimera.cli rag <corpus.json> --query="..." [--json]
python -m chimera.cli parse  <file>          # Print AST
python -m chimera.cli lex    <file>          # Print token stream
python -m chimera.cli repl                   # Interactive REPL
```

---

## Hallucination-Guarded RAG

ChimeraLang includes a local RAG runtime for grounded answers with citations and guard results:

```bash
python -m chimera.cli rag examples/rag_corpus.json --query="How does ChimeraLang ground RAG answers?" --json
```

The corpus is a JSON array of documents:

```json
[
  {
    "id": "retrieval",
    "text": "RAG answers should cite retrieved documents.",
    "metadata": {"source": "runtime"}
  }
]
```

The runtime uses deterministic hashing embeddings, `VectorStore` retrieval, extractive answer synthesis, `GuardLayer` confidence/variance checks, and `ConstitutionLayer` safety checks. If retrieval is weak, the answer is refused instead of hallucinated.

---

## Verifiable Certificates

A ChimeraLang program can emit a **portable certificate** of its own reasoning that any third party can verify **offline**, with nothing but the certificate file. The verifier (`chimera/verify.py`) imports only the Python standard library and nothing from the execution path — it re-derives every hash from the certificate itself.

**Emit a certificate** alongside the human-readable integrity report:

```bash
# Tamper-evident certificate (standard library only)
python -m chimera.cli prove examples/belief_reasoning.chimera --out=cert.json

# Add shared-secret authentication (HMAC-SHA256)
python -m chimera.cli prove examples/belief_reasoning.chimera --out=cert.json --key=hmac.key

# Add an asymmetric, third-party-verifiable signature (needs `cryptography`)
python -m chimera.cli prove examples/belief_reasoning.chimera --out=cert.json --sign-key=ed25519.pem
```

**Verify a certificate** (exit code `0` = verified, `1` = failed):

```bash
python -m chimera.cli verify cert.json                  # tamper-evidence
python -m chimera.cli verify cert.json --key=hmac.key   # + authenticate with shared secret
python -m chimera.cli verify cert.json --pubkey=HEX     # + verify signature against a trusted key
```

The verifier runs **all** checks and reports **every** failure (no early exit): certificate-hash binding, optional HMAC, reasoning-chain integrity, gate-certificate hashes, verdict consistency, and the optional signature.

### Guarantees (stated exactly — no stronger claims)

| Mechanism | What it proves | Dependency |
|---|---|---|
| **Hash binding** (`certificate_hash`, SHA-256) | **Tamper-evidence** — binds every report field to one digest, so corruption or modification is caught when the expected digest is known through a trusted channel. | stdlib |
| **HMAC-SHA256** (`--key`) | **Authentication via a shared secret** — confirms the holder of the secret produced the certificate. | stdlib |
| **Ed25519 signature** (`--sign-key` / `--pubkey`) | **Asymmetric, third-party-verifiable signature** over the canonical report. | optional `cryptography` |

**On the bare hash binding:** the `certificate_hash` travels inside the certificate, so by itself it detects accidental corruption and lets you pin/compare a known-good digest out-of-band — it does **not** stop a motivated adversary, who can edit the report and recompute the digest. For authentication against untrusted parties, use HMAC (shared secret) or Ed25519 (asymmetric signature).

The Ed25519 layer is **optional**: signing requires `pip install cryptography`, and if it is absent, every other feature still works. When verifying a signed certificate **without** providing a trusted `--pubkey`, the signature is checked only against the certificate's **own embedded** public key. That is a **trust-on-first-use self-consistency check, not proof of authorship** — anyone can mint a key and embed it. Authorship is established only by verifying against a `--pubkey` you already trust through an independent channel.

---

## Static Capability Enforcement

`fn` declarations can declare capability constraints with `allow` and `forbidden`. These are now **statically enforced**: the type checker infers which capability-bearing operations each declaration uses — transitively through calls to other declarations — and rejects a program whose body violates its own declared constraints, **before it runs**. This is the compile-time half of the verifiable-by-construction property; the runtime half ships in the certificate layer above.

```bash
python -m chimera.cli check <file.chimera>                  # type + capability check (exit 0/1)
python -m chimera.cli run   <file.chimera>                  # refuses to execute a violating program
python -m chimera.cli run   <file.chimera> --no-capability-check   # downgrade capability violations to warnings
```

**Capabilities** are grounded in operations that actually exist in the AST/adapter:

| Operation | Capabilities | Where |
|---|---|---|
| Agent inquiry (`belief x := inquire { agents: [...] }`) | `model`, `network` | source AST |
| Tool call (`ToolCallSpec` via the Claude adapter) | `tool`, `network` | host adapter |
| `print` builtin (console output) | `io` | source AST |
| All other builtins (`confident`, `consensus`, `len`, …) | none (pure) | — |

**Semantics:** `forbidden c` makes any use of capability `c` an error. When an `allow` clause is present it is a **whitelist** — any used capability not listed is an error; with no `allow` clause, every non-forbidden capability is permitted. Only the canonical names (`network`, `model`, `filesystem`, `tool`, `io`) are treated as capabilities; other `allow`/`forbidden` strings (e.g. `"external tool invocation"`) remain free-form semantic annotations and are ignored by the capability checker. `must:` constraints continue to be enforced at runtime as before.

When a certificate is produced (`prove --out`), its full report carries a `capabilities` block attesting that static checking ran at prove time, plus the `declared`/`used` capability sets per declaration. This attests only that **the static check passed when the certificate was produced** — it is covered by the existing certificate hash and re-verified with no verifier change.

**Guarantee, stated exactly:** declared capability constraints are enforced against *statically known* capability-bearing operations (agent inquiries, tool calls, `print`). It is **not** a runtime sandbox and not a proof of runtime isolation — it cannot constrain effects the type checker cannot see (e.g. capabilities introduced by host code or by operations the language does not yet model).

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
│   ├── vm.py                 # Ensemble-gate VM (fn/gate/goal/reason path)
│   ├── detect.py             # Hallucination detector
│   ├── integrity.py          # Merkle chains & gate certificates
│   └── cli.py                # CLI + REPL + automatic CIR/VM dispatch
├── examples/
├── spec/SPEC.md
├── paper/chimeralang.tex
├── tests/                    # 314 tests (310 passing, 4 env-dependent skips: torch, llvm-as)
└── pyproject.toml
```

---

## How It Differs

| Aspect | Traditional Languages | ChimeraLang |
|---|---|---|
| Values | Deterministic | Beta distributions carrying full uncertainty |
| Evidence combination | N/A | Pseudocount addition with conflict check (not naive averaging) |
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
