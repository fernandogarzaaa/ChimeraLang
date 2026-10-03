"""Contract tests for the BetaDist pooling algebra.

Formalization of docs/pooling-redesign.md option (a): each source
contributes its (alpha, beta) pseudocounts as evidence counts, combined
by plain addition with no subtract-one:

    combine(a, b) = Beta(a.alpha + b.alpha, a.beta + b.beta)

after the retained K conflict check.

Justification for this exact formula:

- Every BetaDist in the system has strictly positive parameters
  (from_confidence clips to (1e-6, 1-1e-6) times a positive strength;
  uniform() is (1, 1)), and sums of positives are positive. The raw
  parameters can therefore never be negative, so the old
  max(., 1e-6) clamp path is unreachable and is removed, not kept.
- The design note writes the n-ary pool as Beta(1 + sum alpha,
  1 + sum beta). A per-combination +1 would accumulate k-1 spurious
  priors over k chained sources and would break the arithmetic-mean
  property below, so the single Beta(1,1) prior is NOT added per
  combination. Positivity (hence clamp-unreachability) holds without
  it.
- The pooled mean is then a strength-weighted average of the input
  means: for equal strengths it is exactly the arithmetic mean of the
  confidences, and k identical sources at c < 1 pool to exactly c,
  never 1.0.

Contract:
1. Raw parameters are never negative and no clamp path is reachable
   (seeded property test, confidences up to 0.9999, strengths 2 and 10).
2. k identical sources at confidence c < 1 never pool to a mean of 1.0.
3. The pooled mean lies within [min, max] of the input means.
4. Pooling is commutative and associative.
5. The K conflict check semantics are retained.

RED against the old (alpha+alpha'-1, beta+beta'-1) rule with its
saturation clamp: the old rule fails 1 (subtract-one shifts the sums;
the clamp hides negative raw betas), 2 (saturates to 1.0), and 3
(saturation pushes the mean outside the input range). Items 4 and 5
pin semantics the old rule already had and stay green.
"""
import random

import pytest

from chimera.cir.nodes import BetaDist


def _chain(dists):
    """Left fold of combine_pseudocount over 2+ dists."""
    acc = dists[0]
    for d in dists[1:]:
        acc = acc.combine_pseudocount(d)
    return acc


def _trial_dists(rng, kmin=2, kmax=6):
    """Seeded random sources that never trip the K conflict check.

    All confidences are drawn from one half of [0,1] so pairwise
    K <= 0.5 < 0.8 always; strengths are 2 or 10 per the contract.
    """
    k = rng.randint(kmin, kmax)
    strengths = [rng.choice([2.0, 10.0]) for _ in range(k)]
    lo, hi = (0.5, 0.9999) if rng.random() < 0.5 else (0.0001, 0.5)
    confs = [lo + (hi - lo) * rng.random() for _ in range(k)]
    return [BetaDist.from_confidence(c, strength=s)
            for c, s in zip(confs, strengths)]


class TestPoolingContract:
    def test_parameters_are_plain_sums(self):
        """Contract 1: pooled params equal the sums of the inputs.

        No subtract-one, no clamp: the formula itself is the proof that
        raw parameters stay strictly positive, so no clamp path can be
        reached. RED on the old rule, which returns
        (sum alpha - (k-1), sum beta - (k-1)) clamped at 1e-6.
        """
        rng = random.Random(20261003)
        for _ in range(300):
            dists = _trial_dists(rng)
            pooled = _chain(dists)
            assert pooled.alpha == pytest.approx(sum(d.alpha for d in dists))
            assert pooled.beta == pytest.approx(sum(d.beta for d in dists))
            assert pooled.alpha > 0.0
            assert pooled.beta > 0.0

    def test_no_clamp_on_previously_clamping_inputs(self):
        """Two 0.99 sources at strength 10 drove the old raw beta to
        0.1 + 0.1 - 1.0 = -0.8 and the max(., 1e-6) clamp hid it.
        The new rule returns the exact sum, beta = 0.2."""
        a = BetaDist.from_confidence(0.99, strength=10.0)
        b = BetaDist.from_confidence(0.99, strength=10.0)
        pooled = a.combine_pseudocount(b)
        assert pooled.alpha == pytest.approx(19.8)
        assert pooled.beta == pytest.approx(0.2)

    def test_identical_sources_never_pool_to_one(self):
        """Contract 2: k identical sources at c < 1 pool to exactly c."""
        for conf in (0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 0.9999):
            for strength in (2.0, 10.0):
                for k in (2, 3, 5, 10):
                    dists = [BetaDist.from_confidence(conf, strength=strength)] * k
                    pooled = _chain(dists)
                    assert pooled.mean < 1.0, (
                        f"c={conf} strength={strength} k={k} pooled to {pooled.mean}"
                    )
                    assert pooled.mean == pytest.approx(conf)

    def test_pooled_mean_within_input_range(self):
        """Contract 3: pooled mean is a strength-weighted average of the
        input means, hence within [min, max]. RED on the old rule: the
        saturation clamp pushes the mean above the inputs' max."""
        rng = random.Random(20261004)
        for _ in range(300):
            dists = _trial_dists(rng)
            pooled = _chain(dists)
            means = [d.mean for d in dists]
            assert min(means) - 1e-9 <= pooled.mean <= max(means) + 1e-9, (
                f"pooled mean {pooled.mean} outside "
                f"[{min(means)}, {max(means)}]"
            )

    def test_commutative(self):
        """Contract 4a: a (+) b == b (+) a exactly."""
        rng = random.Random(20261005)
        for _ in range(100):
            a = BetaDist.from_confidence(rng.uniform(0.6, 0.9), strength=10.0)
            b = BetaDist.from_confidence(rng.uniform(0.6, 0.9), strength=10.0)
            ab = a.combine_pseudocount(b)
            ba = b.combine_pseudocount(a)
            assert (ab.alpha, ab.beta) == (ba.alpha, ba.beta)

    def test_associative(self):
        """Contract 4b: (a (+) b) (+) c == a (+) (b (+) c).

        Float addition is not bit-associative, so approx, not exact.
        """
        rng = random.Random(20261006)
        for _ in range(100):
            ds = [BetaDist.from_confidence(rng.uniform(0.6, 0.9), strength=10.0)
                  for _ in range(3)]
            left = ds[0].combine_pseudocount(ds[1]).combine_pseudocount(ds[2])
            right = ds[0].combine_pseudocount(ds[1].combine_pseudocount(ds[2]))
            assert left.alpha == pytest.approx(right.alpha)
            assert left.beta == pytest.approx(right.beta)

    def test_conflict_check_retained(self):
        """Contract 5: irreconcilable sources still raise ValueError."""
        a = BetaDist(alpha=19.0, beta=1.0)
        b = BetaDist(alpha=1.0, beta=19.0)
        with pytest.raises(ValueError, match="conflict"):
            a.combine_pseudocount(b, conflict_threshold=0.5)

    def test_no_conflict_below_threshold_combines(self):
        a = BetaDist.from_confidence(0.9, strength=10.0)
        b = BetaDist.from_confidence(0.6, strength=10.0)
        pooled = a.combine_pseudocount(b)  # must not raise
        assert pooled.mean == pytest.approx(15.0 / 20.0)
