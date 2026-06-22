# Changelog

All notable changes to ChimeraLang are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
