# Guard dominance: design and implementation (2026-10-03)

**Status: implemented** on branch `feat/guard-dominance` (PR pending).
The static check, score provenance, v2 certificates, and independent
verification described below are implemented and tested
(`tests/test_guard_dominance.py`, 25 tests). The current state it
builds on is mapped in
`docs/design/guard-dominance-current-state.md`. All six product
decisions are marked CHOSEN below.

**Update 2026-10-03 (Option 2 implemented):** the source-level strict
guard modifier is now implemented on branch `feat/strict-guard-source`.
`guard x against hallucination { ..., strict: true }` is parsed into
`GuardStmt.strict`, lowered to `ValidationNode.strict`, and honored by
the executor (a failing strict guard raises `GuardViolation` even when
the global `--strict-guard` flag is off). The dominance claim is now
`enforced` only when every guard on every dominating path to each
effectful node is source-level strict; the global `strict_guard` run
flag is recorded in the certificate for information only and does not
affect the claim. The verifier derives the claim from the re-lowered
graph alone and ignores `cir.strict_guard`. See
`examples/strict_guarded_pipeline.chimera`, which verifies as
`enforced` via `chimera verify`.

## 0. The check in one paragraph

Every effectful CIR node must be guard-dominated along belief-flow
edges: for each belief it consumes, every belief-flow path from a
source to the node must pass through a `ValidationNode` that validated
that same belief. The check runs at lowering time (a new pass in
`CIRLowering`, after `_pass_structural`), rejects violating programs
with a `LoweringError` naming the node and belief, and records the
dominance evidence in a new CIR certificate section that
`chimera/verify.py` recomputes from the graph structure instead of
trusting.

## 1. Definitions

### 1.1 Effectful

Grounded in what the nodes actually do (`chimera/cir/executor.py`):

- `EvolutionNode`: **effectful**. `_exec_evolution` re-invokes the
  inquiry adapter (network model call) up to `max_iter` times and
  rewrites the belief distribution from fresh responses.
- `emit` (via `CIRGraph.emit_ids`): **effectful**. It is not a node,
  but it is the program's observable output: the belief leaves the
  program through `CIRResult.emitted` (printed by `chimera/cli.py`,
  returned by `run_cir`).
- `InquiryNode`: effectful (network model call per agent,
  `_exec_inquiry`), but it is a belief **source** with no belief input,
  so the dominance requirement is vacuous for it. It is constrained by
  the existing capability system (`chimera/capabilities.py`), not by
  this check.
- `ConsensusNode`, `ValidationNode`, `MetaNode`: **pure**. Local
  computation only; no network, no output.

### 1.2 Dominated

Let G be the lowered `CIRGraph`. Belief-flow edges are the `CIREdge`s
created by lowering (`EdgeKind.CONSENSUS`, `EdgeKind.VALIDATION`,
`EdgeKind.EVOLUTION`; see current-state doc section 2).

For a belief name b, define its **node lineage** as the sequence of node
ids that `belief_node_map[b]` takes during `_pass_structural`
(`chimera/cir/lower.py`): inquiry ids, then `[consensus.id]` after
`resolve`, then `[validation.id]` after `guard`, then `[evolution.id]`
after `evolve`. For single-source beliefs this is a linear chain; for
fan-out beliefs each agent (`x@a`, `x@b`) has its own chain until
`resolve` collapses them.

An effectful consumer E of belief b is **guard-dominated** iff, for
every agent-chain of b that reaches E, the chain contains a
`ValidationNode` V positioned **before** E whose `target_id` is the
node id that b mapped to at the `guard` declaration (i.e., V validated
exactly the belief version E consumes, not an ancestor version).

Concretely:
- `inquire -> guard -> emit`: dominated (chain I, V; emit consumes V's belief).
- `inquire -> resolve -> guard -> evolve -> emit`: dominated for the
  evolve (V before E) and for the emit (V before emit in the lineage).
- `inquire -> evolve -> emit`: **not** dominated (no V in the chain).
- `inquire -> evolve -> guard -> emit`: the evolve is **not** dominated
  (V comes after E); the emit is dominated.

### 1.3 Same belief across consensus and evolve rewrites

Two mechanisms, one at lowering time (used by the static check) and
one at execution time (used by the certificate):

- **Lowering time**: `belief_node_map` threading in
  `_pass_structural`. After `resolve`, the name maps to the consensus
  id; after `guard`, to the validation id; after `evolve`, to the
  evolution id. "The same belief" for the static check means the same
  belief **name** traced through these rewrites.
- **Execution time**: `BeliefState.node_id` rewrites. Consensus
  rewrites it (`executor.py:381`), validation rewrites it on pass
  (`executor.py:442`), evolve rewrites it (`executor.py:515`). The
  certificate records, per emitted belief, the ordered list of node ids
  its `BeliefState.node_id` took, so the verifier can match the static
  lineage against the runtime lineage.

A validation dominates only the exact belief version it checked: a V
whose `target_id` is the consensus id dominates the consensus belief,
not the pre-consensus inquiry beliefs.

## 2. Uncalibrated posteriors at numeric guard thresholds

The guard today judges `P(correct)` as `bs.calibrated_p` when a
calibrator was supplied, else the uncalibrated posterior mean
(`executor.py:_exec_validation`: `score = bs.calibrated_p if
bs.calibrated_p is not None else dist.mean`).

The H1 confirmatory results show this fallback is unsound as a
probability (`experiments/h1_pooling/PREREG_V2_CONFIRM.md:188-189`):
the uncalibrated posterior-mean Brier was 0.2134 in mode A vs constant
0.2183, and **0.2980 in mode B vs constant 0.2176**. In mode B the
uncalibrated score is worse than predicting the base rate, so a guard
threshold like `max_risk: 0.2` (requiring score >= 0.80) evaluated on
an uncalibrated mean is not a real 0.80 gate. Calibration is
load-bearing, and the confirmatory verdict on calibration was
Inconclusive (mode-A CI [-0.0007, 0.0300] includes zero), so even the
calibrated path deserves skepticism.

Options (product decision required):

- **Option A (strict)**: a guard with `strategy` in `("mean", "both")`
  is a lowering error unless a calibrator is supplied to `run_cir`
  (or `--calibrator=PATH` on the CLI). Uncalibrated programs can still
  use `strategy: "variance"`, which does not pretend to be a
  probability.
- **Option B (attest)**: keep the fallback, but record
  `score_source: "calibrated" | "uncalibrated"` on the validation
  trace entry and in the certificate (section 3), so the verifier and
  any downstream policy can distinguish a real probability gate from a
  heuristic one. The lowering warning about the missing calibrator
  (already emitted by `run_cir` in `chimera/cir/__init__.py`) becomes
  load-bearing documentation.
- **Option C (cap)**: uncalibrated means may satisfy only weak
  thresholds (e.g., required score <= 0.60); anything stricter without
  a calibrator is a lowering error. The 0.60 cap is arbitrary and
  would need its own justification.

Recommendation: Option A for `mean`/`both` guards combined with Option
B's attestation everywhere, so that even variance-only guards carry
their score provenance. This is a recommendation only; the decision is
open.

## 3. Certificate attestation and independent verification

There is no CIR certificate today: `IntegrityEngine.certify` takes a
VM-path `ExecutionResult` (`chimera/integrity.py`), and `run_cir`
results are never certified (current-state doc section 4). Dominance
attestation is therefore new machinery, not an extension:

- **Producer** (new code, not this doc): the lowering pass that checks
  dominance records, per emitted belief, the dominance evidence: the
  ordered node-id lineage, and for each `ValidationNode` in it the
  `strategy`, `max_risk`, `max_variance`, and `score_source`
  (`calibrated`/`uncalibrated`, section 2). This evidence is embedded
  in a new CIR certificate section alongside the lowered graph
  structure needed to check it (node kinds, edges, validation
  parameters). Certificate format version bumps (e.g.
  `chimeralang-cert/v2` or a `cir` top-level section); the v1 VM-path
  format is untouched.
- **Verifier** (`chimera/verify.py`, extended): recomputes the
  dominance predicate **from the embedded graph structure**, not from
  any stored boolean. Concretely it re-derives, for each emitted
  belief, the agent-chains from the edge list, checks that every chain
  to each effectful consumer passes through a `ValidationNode` whose
  `target_id` matches the lineage position, and fails the certificate
  if any chain does not. A stored `dominated: true` flag with no
  accompanying graph structure is treated as absent evidence and
  fails. This mirrors how `verify.py` already recomputes chain hashes,
  gate hashes, and the verdict rather than trusting stored values.

**Trust boundary, stated exactly.** The v2 verifier additionally
re-lowers `cir.program_source` with the real parser and lowering and
compares the canonical serialized graph byte for byte to the embedded
graph (node ids are deterministic creation-order ids assigned by
`CIRLowering`); a mismatch is a failure, which defeats a producer that
splices in a graph from a different program and fixes up the hashes.
When the chimera package cannot be imported, the graph-source link is
NOT RE-DERIVED: the verifier checks internal consistency only, reports
the link status, and never reports valid for a certificate claiming
`enforced`. `chimera verify` prints the link status (`RE-DERIVED` /
`NOT RE-DERIVED` / `N/A` for v1).

### strict_guard is producer-asserted, not source-derivable

The `strict_guard` flag in the v2 certificate is asserted by the
producer: it records the `strict_guard` argument passed to `run_cir`
(or `--strict-guard` on the CLI). It is not derivable from
`cir.program_source`, because the source language has no strictness
modifier; strictness is a run-time executor option. The verifier's
re-derivation therefore cannot independently confirm that the run
actually executed with strict guards on. It can only confirm that
*if* the producer's assertion is true, the `enforced` claim follows
from (dominance holds AND strict_guard asserted).

This is a deliberate limitation, not a bug, but the word `enforced`
overstates what the certificate proves. Two options, neither
implemented yet (the claim is unchanged):

**Option 1: rename the claim.** Change `enforced` to a name that
marks the producer assertion, e.g. `dominated-strict-asserted`.
- Syntax: none.
- Lowering: none (the dominance pass is unchanged).
- Verifier: expect the new claim string in
  `_expected_dominance_claim`; the predicate (dominated AND asserted
  strict_guard) is unchanged. Update `certify.py:dominance_claim`,
  the CLI output text, and docs.

**Option 2: source-level strict guard modifier.** Add strictness to
the language, e.g. `guard x against hallucination { max_risk: 0.2,
strict: true }`, so the flag is part of the program source and thus
re-derivable by the verifier.
- Syntax: new optional `strict` field on `guard_stmt` in
  `docs/grammar.ebnf`; parser support in `_parse_guard`.
- Lowering: store `strict` on `ValidationNode`; the executor's
  `_exec_validation` consults the per-guard flag (threaded from the
  node) in addition to or instead of the run-time `strict_guard`
  argument. The dominance pass is unchanged.
- Verifier: the strict flag is now in the re-derived graph, so the
  `enforced` claim becomes fully source-derivable with no producer
  assertion; `certify_cir` reads it from the graph instead of taking
  a `strict_guard` argument.

## 4. Red tests

These are specifications for tests to be written when the check is
implemented. Programs are in the CIR surface syntax (cf.
`docs/superpowers/plans/2026-04-27-cir-belief-system.md`).

**Must be rejected** (lowering error naming the node and belief):

1. Emit without guard:
   `belief x := inquire {...}` then `emit x`.
2. Evolve without guard (the current-state doc's proof program):
   `belief x := inquire {...}`, `evolve x until stable { max_iter: 3 }`,
   `emit x`.
3. Guard after evolve (evolve undominated):
   `belief x := inquire {...}`, `evolve x until stable { max_iter: 3 }`,
   `guard x against hallucination { max_risk: 0.2 }`, `emit x`.
   The evolve must be rejected even though a later guard exists.
4. Fan-out resolve without guard before evolve:
   `belief x := inquire { agents: [a, b] }`,
   `resolve x with consensus { threshold: 0.8 }`,
   `evolve x until stable { max_iter: 3 }`, `emit x`.
   The evolve consumes the consensus belief with no validation.
5. Uncalibrated mean guard under Option A (section 2):
   `belief x := inquire {...}`,
   `guard x against hallucination { max_risk: 0.2, strategy: mean }`,
   `emit x`, run with no calibrator. Lowering error.

**Must pass:**

1. Canonical: `inquire`, `resolve`, `guard`, `evolve`, `emit` in order.
2. Single-source: `belief x := inquire {...}`,
   `guard x against hallucination { max_risk: 0.2 }`, `emit x`.
3. Guard before evolve, single source:
   `inquire`, `guard`, `evolve`, `emit` (dominated; see falsifiers for
   why this is still weak).

**Edge cases:**

- Fan-out with resolve then guard then emit: dominated via the
  consensus belief; every agent chain passes through the validation.
- `dempster_shafer` strategy: lowering rewrites it to `pooled` with a
  warning (`lower.py:_pass_structural`); dominance is unaffected
  because the check keys on the `ConsensusNode`, not the strategy
  string. A test must assert the alias path still dominates.
- Guard with explicit `max_variance`: dominates like any guard; the
  check does not re-evaluate the variance math, only the presence and
  position of the validation.
- Double guard: `inquire`, `guard`, `guard`, `emit`. Passes; the second
  guard is redundant but harmless.
- `MetaNode` introspection: pure, never subject to the check, never
  satisfies it for another node.

## 5. Falsifiers

Cases where the check gives **false confidence** (says dominated, but
the guarantee is weaker than it looks):

- **Guard then evolve**: the emitted belief's distribution was
  rewritten by `_exec_evolution` after validation. Static dominance
  holds, but the validated numbers no longer describe the output. The
  certificate's runtime lineage (section 1.3) exposes this; the static
  check alone does not.
- **Non-strict violations**: a dominated emit can still output a
  belief that *failed* its guard, because violations without
  `strict_guard=True` only append to `result.guard_violations` and
  execution continues (`executor.py:436`). Dominance is about the
  presence of a check, not about its outcome.
- **Seeded priors**: `SymbolStore.find_prior_for` seeds the prior from
  past observations (`lower.py:_pass_structural`). The guard validates
  the posterior, but the prior's provenance bypasses fresh validation;
  a poisoned or stale symbol weakens what dominance means.
- **Calibrator staleness**: `calibrated_p` is only as good as the
  held-out fit. The confirmatory calibrators were fit on questions
  1-500 and applied to 501-1000; under domain shift the "calibrated"
  score is miscalibrated, and the guard cannot tell.

Cases where the check **rejects safe programs**:

- **Emit of an introspection result**: a `MetaNode` reading a guarded
  belief produces trace text, not a belief; if a future surface lets
  programs emit introspection output, the check has no belief lineage
  to evaluate and would reject. (No such surface exists today:
  `MetaNode` is never constructed by lowering.)
- **Guard placement vs. pure rewrites**: `inquire`, `guard`,
  `resolve`, `emit` (guard before resolve). The consensus belief was
  never directly validated, so the check rejects the emit, even
  though every source was validated and agreement only ever lowers
  the claimed confidence relative to the sources. Whether this
  conservatism is acceptable is a product decision.
- **Evolve with `max_iter: 0` or a no-op condition**: still rejected
  statically, because the check cannot see that the evolution would
  not change the belief.

## 6. Public behavior changes and backward compatibility

- **New lowering error**: programs that today lower cleanly and run
  (any `emit` or `evolve` not preceded by a `guard` on the same
  belief, as proven in the current-state doc section 5) would fail at
  lowering time. This is a breaking change for existing programs,
  including possibly programs under `examples/` and
  `experiments/h1_pooling/fixture/`; the audit of which existing
  programs would newly fail has **not** been done (listed under
  unverified below).
- **Error vs warning**: the check could ship as a `LoweringError`
  (hard break) or as a lowering warning with an opt-in strict flag
  (soft rollout). The warning path matches the existing style
  (`_first_source_or_warn`, the uncalibrated-posterior warning), but a
  warning that is routinely ignored provides no guarantee; the
  certificate would then need to attest "dominance: warning-only".
- **`strict_guard` interaction**: none by default. `strict_guard`
  (`CIRExecutor.__init__`, `executor.py:188`) controls whether a
  *failed* guard raises at runtime; dominance is a static property
  about whether a guard *exists* on the path. They compose: dominance
  without `strict_guard` still permits emitting guard-failed beliefs
  (falsifier above). A future option could tie them together
  (dominance required *and* strict), but that is not proposed here.
- **CLI surface**: `chimera run` would surface the new lowering error
  like any `LoweringError`; no CLI flag changes are proposed in this
  doc.
- **Versioning**: the certificate format bump (section 3) is additive
  for the VM path; old v1 certificates still verify.

## 7. Related work

Only sources actually found via web search are listed, with verbatim
URLs. Every novelty claim below is marked **unverified**.

- **Dominator trees** (the static-analysis concept this proposal
  borrows its name from): a node d dominates n iff every path from the
  entry to n passes through d (Cytron et al., "Efficiently Computing
  Static Single Assignment Form and the Control Dependence Graph",
  TOPLAS 1991). Introductory treatment:
  https://www.cs.cornell.edu/courses/cs6120/2020fa/lesson/5/ . The
  proposal applies the same definition to belief-flow edges instead of
  control-flow edges. **Unverified**: whether "guard dominance on a
  belief-flow graph" as formulated here (per-belief lineages, validation
  nodes as dominators, effectful consumers as dominated nodes) appears
  in prior literature; no prior-art search beyond the sources listed
  here was performed.
- **Jif / decentralized label model** (Myers and Liskov): static
  checking of information-flow policies, where security labels on data
  are enforced at compile time so that programs can be certified to
  permit only acceptable flows:
  http://www.scs.stanford.edu/nyu/05sp/sched/readings/jif.pdf . Jif
  enforces *which flows are allowed*; guard dominance enforces *which
  checks must precede effects*. Related in spirit (static certification
  of a safety property), different property.
- **Koka effect system** (Daan Leijen, Microsoft Research):
  row-polymorphic effect types that statically track which effects a
  function may perform:
  https://koka-lang.github.io/koka/doc/book.html . Koka tracks *what
  effects occur*; guard dominance additionally constrains *what must
  happen before* specific effects on specific data. The `effectful`
  definition in section 1.1 is analogous to an effect row with two
  labels (model re-query, belief output).

## Product decisions (CHOSEN)

1. CHOSEN: lowering warning by default; `run_cir(require_dominance=True)`
   and CLI `--require-dominance` make it a `LoweringError`.
2. CHOSEN: Option A for `mean`/`both` guards combined with Option B's
   attestation everywhere. Under `require_dominance`, a `mean`/`both`
   guard with no calibrator is a `LoweringError`; `score_source`
   (`calibrated`/`uncalibrated`) is recorded on every validation trace
   entry and in the certificate.
3. CHOSEN: guard-before-resolve is not accepted as dominating the
   consensus. Documented as intentional conservatism.
4. CHOSEN (superseded by Option 2): `strict_guard` was recorded in the
   certificate, and the claim was `enforced` only when dominance held
   AND `strict_guard` was on. **Option 2 is now implemented:** the
   claim is `enforced` only when every guard on every dominating path
   is source-level `strict: true`. The run flag is recorded for
   information only and ignored by the verifier.
5. CHOSEN: new `chimeralang-cert/v2` envelope with a `cir` section.
   `verify.py` fails closed on unknown versions and verifies v1
   unchanged.
6. CHOSEN: effectful nodes are `EvolutionNode` and `emit`
   (`InquiryNode` is a source, so the requirement is vacuous for it).
7. CHOSEN: guard strength. The `enforced` claim is structural only.
   Vacuous guards (those that can never fail) are detected via
   `is_vacuous_guard()`; lowering warns, `--require-dominance` raises
   `LoweringError`. The certificate lists each dominating guard with
   its `vacuous` flag, and `guard_strength` is `vacuous`,
   `uncalibrated`, or `nonvacuous` (worst case). The verifier
   recomputes the list and rejects tampering.

### Vacuous guard definition

A guard is vacuous iff none of its active checks can fail:

- Mean check (`strategy` "mean" or "both"): violation if
  `score < 1.0 - max_risk`. Scores are Beta means in (0, 1]. If
  `max_risk >= 1.0`, the threshold is <= 0, which no score can fall
  below.
- Variance check (`strategy` "variance" or "both"): violation if
  `variance > max_variance`. Inquiry-produced beliefs have variance <=
  `BetaDist.max_variance_for_strength(10.0)` ≈ 0.0227. An explicit
  `max_variance` at or above this cap can never fire.

### False-confidence falsifier

**Claim:** A strict guard with `max_risk: 1.0` yields "enforced" but
can never fail.

**Reproduction:**
```
belief x := inquire { prompt: "Is the sky blue?", agents: [claude] }
resolve x with consensus { threshold: 0.8 }
guard x against hallucination { max_risk: 1.0, strategy: mean, strict: true }
emit x
```
Lowering warns: "guard on 'x' is vacuous: it can never fail".
The certificate claims "enforced" (structural dominance holds) but
`guard_strength` is "vacuous" and `chimera verify` prints:
"WARNING: 1 dominating guard(s) are vacuous (can never fail); the
'enforced' claim is structural only."

### Explicit non-claims

The "enforced" claim does NOT mean:
- The guards are meaningful (a vacuous guard still yields "enforced")
- The guards are calibrated (uncalibrated posterior means have
  AUROC 0.67-0.75, not perfect)
- A particular run was honest
- The beliefs were well-formed
- Any specific model produced the beliefs

It means: the source has the structural property that every
effectful node is dominated by source-level strict guards.

## What was verified, and what was not

Verified during implementation:
- The dominance audit: 11 `.chimera` files across examples/, tests/,
  experiments/ scanned; 0 undominated effectful nodes. Only
  `examples/belief_reasoning.chimera` uses the CIR belief surface.
- Certificate sizing: embedded graph section ~1.5KB (10 nodes),
  ~12KB (100 nodes), ~121KB (1000 nodes); dominance recompute
  0.07ms/0.33ms/3.34ms. `verify.py` stays stdlib-only at import time.
- The lowering pass runs between `_pass_structural` and
  `_pass_dead_belief_elimination` (`lower.py`).
- Graph-source forgery: a red test splices a dominated graph into an
  unguarded program's certificate with all hashes fixed up; the
  verifier rejected it only after re-derivation was added
  (`test_verifier_rejects_forged_graph_source_mismatch`).
- Differential test: lowering, certify, and verifier dominance
  implementations agree on all examples, fixture programs, and 200
  seeded random programs (196 with effectful consumers).
- NOT RE-DERIVED path: simulated missing chimera import; link status
  reported, `enforced` claim never valid
  (`test_not_rederived_never_valid_with_enforced`).

Not verified:
- Any prior art beyond the three sources in section 7; all novelty
  claims remain marked unverified.
- The verifier's behavior on adversarially large embedded graphs
  (no fuzzing run).
- Whether `belief_reasoning.chimera`'s `strategy: both` guard should
  be changed to `variance` or given a checked-in calibrator so it
  passes `--require-dominance` out of the box (left unmodified).
