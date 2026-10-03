# Changelog

All notable changes to ChimeraLang are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Static guard-dominance check.** New `_pass_guard_dominance` lowering
  pass (between structural and dead-belief elimination): every
  effectful consumer (`evolve` node, `emit` target) must be dominated
  by a validation node on its belief lineage, positioned after any
  `resolve`. Default is a lowering warning; `run_cir(...,
  require_dominance=True)` and CLI `--require-dominance` make it a
  `LoweringError`. Under `require_dominance`, `mean`/`both` guards
  without a calibrator are also `LoweringError`s.
- **Guard score provenance.** Every guard trace entry records
  `score_source` (`calibrated` when `calibrated_p` was judged, else
  `uncalibrated`), and `CIRResult.validations` carries the structured
  per-guard record.
- **CIR certificates (chimeralang-cert/v2).** `chimera.cir.certify.certify_cir`
  builds a v2 envelope with a `cir` section (program source/hash,
  lowered graph/hash, dominance claim `enforced`/`non-blocking`/`absent`,
  validations with `score_source`, `strict_guard` flag). CLI
  `--cert-out=PATH` writes one; `chimera verify` recomputes dominance
  from the embedded graph via an independent stdlib-only
  implementation and never trusts the stored claim. Unknown versions
  fail closed; v1 verification is unchanged.
- **Example:** `examples/guarded_pipeline.chimera`, a canonical
  guard-dominated pipeline passing `--require-dominance`.

- **Agreement resolve strategy with opt-in calibration.** `resolve` now
  accepts `strategy: agreement` (the default when a belief has more
  than one source with answers), `strategy: pooled` (the Stage 2 Beta
  algebra), and `strategy: dempster_shafer` (accepted as an alias of
  `pooled` with a lowering warning). Agreement voting
  (`chimera/cir/agreement.py`) counts normalized answers (same
  normalization as the H1 experiment), winner is the most common with
  ties broken by earliest source in agent order, and the resolved
  belief carries raw agreement (votes/N), a Laplace-smoothed Beta
  posterior (unanimous 3/3 gives Beta(4,1), mean 0.8, never 1.0), and
  the winning answer text. The comparator and normalizer are pluggable.
  `chimera/cir/calibration.py` adds `LogisticCalibrator`: deterministic
  pure-Python logistic fit, JSON round-trip carrying n, dataset hash,
  and fit date, refusing fewer than `MIN_FIT_N = 30` points.
  `run_cir` accepts `calibrator=...`; the CLI accepts
  `--calibrator=PATH`. `calibrated_p` appears on resolved beliefs only
  when a calibrator is supplied; guard and emit use it when present,
  otherwise the uncalibrated posterior with a lowering warning.
  Behavior change: the default resolve strategy changed from
  `dempster_shafer` to `agreement`; multi-source beliefs with answers
  now resolve by vote share instead of Beta pooling unless
  `strategy: pooled` is given explicitly.

- **CIR fan-out: one `InquiryNode` per agent.** A belief with
  `agents: [a, b]` now lowers to two `InquiryNode`s (one per agent) and
  `resolve` pools them with the shipped `combine_pseudocount` chain.
  Each agent's belief is tracked as `x@a`, `x@b` in the belief store.
  Behavior change: multi-agent beliefs now cause N adapter calls, one
  per agent. Different agents are less correlated, not independent
  (models share training data), so pooling still overstates the
  evidence somewhat. Without `resolve`, `guard`/`evolve`/`emit` apply
  to the first agent's belief only and lowering warns loudly.
  Duplicate agent names warn that the sources are the same model and
  the evidence is correlated. The default Anthropic adapter maps agent
  names to model ids through an explicit `agent_models` dict (default
  `{"claude": "claude-sonnet-4-6"}`) and raises a clear error naming
  any unknown agent instead of silently falling back to one model.

### Fixed

- **Pooling algebra no longer saturates.** `BetaDist.combine_pseudocount`
  (and its alias `combine_ds`) now adds evidence counts directly,
  `(alpha + alpha', beta + beta')`, with the K conflict check retained.
  The old rule `(alpha + alpha' - 1, beta + beta' - 1)` with its
  `max(., 1e-6)` clamp drove the raw beta to zero and negative for
  agreeing high-confidence sources, so the pooled mean saturated at
  exactly 1.0 (e.g. five 0.95 sources at strength 10). The new rule
  cannot clamp: parameters are sums of strictly positive
  pseudocounts. Behavior change: pooled means are now strength-weighted
  averages of the input means (exactly the arithmetic mean for equal
  strengths); `Beta(8,2) + Beta(8,2)` now pools to `Beta(16,4)` with
  mean 0.8 instead of `Beta(15,3)` with mean 0.8333. Downstream
  thresholds on combined means fire differently (less extreme).
  `_exec_inquiry`'s seeded-prior path used the same old formula inline;
  it now uses `combine_pseudocount`, and a conflicting seeded prior is
  recorded as a guard violation instead of being silently merged.

- **Evolve after resolve/guard now re-inquires.** `_exec_evolution` walked only a
  `subgraph_entry` that had to be an `InquiryNode`, so `evolve` after `resolve`
  or `guard` silently ran zero adapter calls and never converged. It now walks
  predecessors back to the originating `InquiryNode` and accepts any belief on
  that chain (consensus/validation rewrite `BeliefState.node_id`).
- **Guard variance limit is now configurable.** `guard ... { max_variance: 0.01 }`
  is parsed, lowered, and enforced (`GuardStmt`, `ValidationNode`, executor).
  Absent, the legacy 0.05 default applies. New lowering warning when a guard's
  variance limit is unreachable: `BetaDist.from_confidence` (strength 10) caps
  variance at about 0.0227, so the default 0.05 check could never fire.
- **Honest evidence combination.** `BetaDist.combine_ds` was documented as
  Dempster-Shafer combination but implements pseudocount addition
  (`alpha + alpha' - 1`, `beta + beta' - 1`) with a K conflict check. The
  implementation is now `combine_pseudocount` with an honest docstring;
  `combine_ds` remains as a backward-compatible alias. Single-source `resolve`
  now traces "single source, no combination performed" instead of a
  misleading "combined mean".
- **Seeded priors are actually used.** `_exec_inquiry` overwrote the
  SymbolStore-seeded prior; it now combines prior and observation by
  pseudocounts (a `Beta(1,1)` prior leaves behavior unchanged). The raw
  observed likelihood is stored on `BeliefState.observed` and only that is fed
  back to the store, fixing double counting of the prior. `Symbol` prior
  strength is capped at the named constant `MAX_PRIOR_STRENGTH = 100.0`
  (proportional rescale, mean preserved).

### Changed

- README belief-reasoning trace, guard/consensus sections, and test count
  updated to match the implementation. Added: pooled beliefs assume
  independent sources; repeated calls to the same model are correlated.

## [0.2.0] - 2026-06-22

This release makes ChimeraLang's reasoning **verifiable by construction**: programs
emit portable, tamper-evident proofs of their own reasoning, and declared capability
constraints are enforced before a program runs.

### Added

- **Verifiable certificates.** `chimera prove --out=cert.json` emits a portable
  `chimeralang-cert/v1` certificate containing the full integrity report
  (Merkle-chained reasoning trace, gate certificates, verdict) bound by a SHA-256
  hash (tamper-evidence). Optional HMAC-SHA256 authentication via `--key`, and
  optional Ed25519 signatures via `--sign-key` (requires the `[sign]` extra).
- **Independent offline verifier.** New `chimera verify <cert.json>` command
  re-checks a certificate without re-executing the program and without importing
  the execution path. It recomputes the hash binding, chain-link hashes, gate
  hashes and the verdict, and (when present) verifies the signature — detecting
  binding, chain, gate, verdict and signature tampering. `--key` authenticates an
  HMAC and `--pubkey=HEX` verifies an Ed25519 signature against a trusted key.
- **Static capability enforcement.** `allow`/`forbidden` constraints on `fn`
  declarations are now enforced at type-check time. The checker infers the
  capabilities each declaration uses (agent inquiries → `model`+`network`, the
  `print` builtin → `io`) transitively through the call graph and rejects
  violations with actionable, site-named errors.
- **`chimera check` command** for type + capability checking (exit `0` clean / `1`
  on error).
- **Certificate capability attestation.** Full reports / certificates carry a
  `capabilities` block (`statically_checked`, plus per-declaration `declared`/`used`
  sets), covered by the existing certificate hash.
- **`sign` packaging extra.** `pip install chimeralang[sign]` pulls in
  `cryptography` to enable Ed25519 signing. Without it, signing degrades gracefully
  and verification of unsigned and HMAC certificates still works.
- **Hallucination-guarded RAG runtime** (`chimera rag`) — JSON-corpus retrieval
  with cited extractive answers, confidence/variance guards, and constitution
  checks; refuses to answer rather than hallucinate when retrieval is weak.

### Changed

- **Execution now gates on static checks.** `chimera run` and `chimera prove`
  refuse to execute (or attest) a program that violates its declared capabilities.
  A `--no-capability-check` escape hatch downgrades *only* capability violations to
  warnings; genuine type errors still block.
- **CIR**: inquiry answers are threaded through the belief pipeline and `BetaDist`
  priors are seeded from the `SymbolStore`.

### Fixed

- `ConstitutionLayer.rounds` now drives an actual critique loop.
- Honest stub markers in roadmap runtime systems; removal of stale `FIX`/`TODO`
  comments; added IR-validity tests.
- Restored `_print_node`, fixed a REPL crash, emitted valid LLVM braces, and made
  the `claude_tool_use` example runnable.

### Guarantees (stated precisely)

- Hash binding = **tamper-evidence** (detects modification; compare the digest
  out-of-band for adversarial assurance). HMAC = **authentication via a shared
  secret**. Ed25519 = **asymmetric, third-party-verifiable signature**. Verifying
  against a certificate's embedded public key is trust-on-first-use, not proof of
  authorship.
- Capability enforcement applies to **statically known** capability-bearing
  operations. It is **not** a runtime sandbox and not a proof of runtime isolation.

## [0.1.0] - 2026-04-29

Initial release.

### Added

- Core ChimeraLang language: probabilistic types (`Confident`/`Explore`/
  `Converge`/`Provisional`), quantum consensus gates, `fn`/`gate`/`goal`/`reason`
  declarations, `for`/`match`, and memory modifiers.
- Cognitive Intermediate Representation (CIR) belief system:
  `belief`/`inquire`/`resolve`/`guard`/`evolve` with Beta-distribution beliefs,
  Dempster-Shafer consensus, free-energy evolution, and symbol emergence.
- Hallucination detection (`detect` blocks and guard nodes).
- Cryptographic integrity reports (`chimera prove`): Merkle-chained reasoning
  traces and gate certificates.
- Compiler backends (`chimera compile`): PyTorch and LLVM IR.
- Interactive REPL (`chimera repl`) and the CLI (`run`/`check`/`lex`/`parse`).

[0.2.0]: https://github.com/fernandogarzaaa/ChimeraLang/releases/tag/v0.2.0
[0.1.0]: https://github.com/fernandogarzaaa/ChimeraLang/releases/tag/v0.1.0
