# Guard dominance: current state (2026-10-03)

Repo-grounded map of what exists today. Every claim cites a file, symbol,
or observed number. This document proposes nothing; the proposal is
`docs/design/guard-dominance.md`.

## 1. CIR node kinds, effectfulness, inputs and outputs

Source: `chimera/cir/nodes.py`. Five node kinds exist.

| node | effectful? | consumes | produces |
|------|-----------|----------|----------|
| `InquiryNode` | **yes** (network: model API call per agent, `executor.py:_exec_inquiry`) | prompt text, agent list | one `BeliefState` per agent (`BetaDist.from_confidence`, strength 10) plus `observed` likelihood |
| `ConsensusNode` | no (pure local math) | `input_ids`: list of source node ids | resolved `BeliefState`: agreement vote-share posterior, or `combine_pseudocount` chain (`executor.py:_exec_consensus`) |
| `ValidationNode` | no (records violations; raises `GuardViolation` only when `CIRExecutor(strict_guard=True)`) | target belief via `target_id` | pass/fail trace entries; on pass rewrites `BeliefState.node_id` to its own id (`executor.py:_exec_validation`) |
| `EvolutionNode` | **yes** (network: re-invokes the inquiry adapter up to `max_iter` times, `executor.py:_exec_evolution`) | belief found by walking `subgraph_entry` back to the originating `InquiryNode` (`_origin_inquiry`) | updated `BeliefState.distribution`, convergence flag |
| `MetaNode` | no (introspection; emits `ReasoningTrace`) | graph | trace |

`emit` is not a node: `CIRGraph.emit_ids: list[str]` names the node ids
whose beliefs are collected into `CIRResult.emitted` at the end of
`CIRExecutor.run` (`executor.py`, emit loop after the topological walk).
The observable effect of a CIR program is this emitted set (printed by
`chimera/cli.py`, returned by `run_cir`).

Effectful therefore means: `InquiryNode` (source of beliefs, network),
`EvolutionNode` (re-query, network), and `emit` (belief escapes the
program as output). `InquiryNode` has no belief input, so a dominance
rule can only constrain `EvolutionNode` and `emit`.

`MetaNode` is declared in `nodes.py` but never constructed by lowering
(`lower.py:_pass_structural` has no `MetaNode` branch).

## 2. Belief-flow edges and acyclicity

Edges are `CIREdge(source_id, target_id, kind)` (`nodes.py`). `EdgeKind`
declares `INQUIRY, CONSENSUS, VALIDATION, EVOLUTION, CAUSAL`, but only
three are ever constructed, all in `lower.py:_pass_structural`:

- `InquiryNode -> ConsensusNode` (`EdgeKind.CONSENSUS`), one per source
- `source -> ValidationNode` (`EdgeKind.VALIDATION`)
- `source -> EvolutionNode` (`EdgeKind.EVOLUTION`)

`INQUIRY` and `CAUSAL` are never constructed anywhere in `chimera/`
(verified by grep). The lowering also threads a `belief_node_map`
(name to node-id list) so that `guard`/`evolve`/`emit` after a `resolve`
see the consensus id, and after a `guard` see the validation id.

The graph is **not** guaranteed acyclic by construction. Lowering never
checks for cycles. `CIRGraph.topological_order` (`nodes.py:300`) uses
Kahn's algorithm and raises `ValueError("CIRGraph contains a cycle")`,
and `CIRExecutor.run` (`executor.py:204`) catches that, records
`converged=False`, and refuses to execute. So a cyclic graph is rejected
at execution time, not prevented at lowering time. In practice the
language surface cannot easily express a cycle (declarations are
sequential and each op consumes the current node id for a belief name),
but nothing in the lowering pass proves it.

## 3. What a ValidationNode attests today

Source: `nodes.py` (`ValidationNode` dataclass) and
`executor.py:_exec_validation`.

Fields: `max_risk: float = 0.2`, `strategy: str = "both"`,
`target_id: str`, `max_variance: float | None = None`.

At execution the guard judges `P(correct)` as `bs.calibrated_p` when a
calibrator was supplied, else the uncalibrated posterior mean (lowering
warns loudly about the missing calibrator). Checks:

- `strategy` in `("mean", "both")`: requires
  `score >= 1.0 - max_risk`.
- `strategy` in `("variance", "both")`: requires
  `variance <= max_variance`, defaulting to the legacy `0.05` when unset.

What it does **not** attest or do:

- It does not record whether the score was calibrated or not in any
  durable form; the trace line says which (`calibrated_p=` vs `mean=`),
  but the `BeliefState` carries no "was calibrated" flag.
- A violation does not stop a non-strict run: it is appended to
  `result.guard_violations` and execution continues (`executor.py`).
  Only `strict_guard=True` raises `GuardViolation`.
- The default variance limit `0.05` can never fire on
  inquiry-produced beliefs (strength 10 caps variance at
  `BetaDist.max_variance_for_strength(10.0)` = 1/44 = 0.0227);
  lowering warns about unreachable limits
  (`lower.py:_pass_belief_flow_analysis`).
- It does not check, and lowering does not check, whether the belief
  reaching it passed through any earlier guard (no dominance notion).

## 4. What the certificate attests today

Sources: `chimera/integrity.py`, `chimera/verify.py`, `chimera/cli.py`.

The certificate path covers the **VM execution path only**.
`IntegrityEngine.certify(exec_result: ExecutionResult, ...)` builds from
`chimera.vm.ExecutionResult`; `run_cir`'s `CIRResult` is never fed into
it, and the `prove` CLI command (`cli.py:320`) runs the VM path
exclusively. There is no CIR-graph certificate today.

What the certificate attests (`integrity.py`):

- `program_hash`: SHA-256 of the source text.
- `reasoning_chain`: Merkle-like SHA-256 chain over the VM execution
  trace strings (`ChainBuilder`).
- `gate_certificates`: per-gate consensus proof (gate name, branch
  count, collapse strategy, branch confidences, result confidence;
  SHA-256 bound in `__post_init__`).
- Assertion pass/fail counts, hallucination flags, static capability
  attestation (`capabilities.py`), and a computed `verdict`.
- Tamper-evidence via `certificate_hash` (SHA-256 over the canonical
  JSON report), optional HMAC-SHA256, optional Ed25519 signature.

What the verifier re-checks vs trusts (`verify.py`, 7 checks):

- Re-checks (recomputes independently): the canonical-JSON
  `certificate_hash` binding, every chain link hash and `prev_hash`
  linkage plus `root_hash`, every gate hash from its fields, the
  verdict from report fields, and the Ed25519/HMAC bindings.
- Trusts (does not re-derive): the semantic content of trace entries
  (the chain proves they are unmodified, not that they are true), the
  gate `result_value` semantics (hash-bound only), and the embedded
  pubkey, which is trust-on-first-use, explicitly not proof of
  authorship (`verify.py` module docstring).

## 5. Nothing prevents guard bypass today: proof

An effectful consumer (`EvolutionNode`, which re-invokes the model
adapter; `emit`, which outputs the belief) can consume a belief that
never passed a `ValidationNode`, and lowering accepts the program
without error.

Minimal program, run through the real parser (`chimera.lexer.Lexer`,
`chimera.parser.Parser`) and the real lowering (`CIRLowering.lower`):

```text
belief x := inquire {
  prompt: "Is the sky blue?",
  agents: [claude]
}

evolve x until stable { max_iter: 3 }

emit x
```

Literal output:

```text
parsed declarations: ['BeliefDecl', 'EvolveStmt', 'EmitStmt']
lowered: 2 nodes, 1 edges
  cdd642a1: InquiryNode
  d6988817: EvolutionNode
  edge InquiryNode -> EvolutionNode [EVOLUTION]
validation nodes present: 0
evolution nodes present: 1
emit ids: ['d6988817']
lowering warnings: []
LOWERING SUCCEEDED WITH NO GUARD IN THE GRAPH
```

The `EvolutionNode` directly consumes the inquiry belief (one
`EVOLUTION` edge, zero `ValidationNode`s, zero warnings), re-queries
the model up to 3 times at execution, and the resulting belief is in
`emit_ids`. The same holds trivially for `emit` without `evolve`:
`emit x` after `inquire` lowers to an emit id on the inquiry node with
no guard involved. Pass 3 (`_pass_belief_flow_analysis`) only warns
about high pre-validation variance and unreachable variance limits; it
has no notion of guard dominance and rejects nothing.
