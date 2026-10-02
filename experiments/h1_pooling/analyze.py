"""Analyze H1 pooling experiment responses.

Reads ONLY responses.jsonl (plus the dataset JSONL for question order and
gold answers). No network. Fully deterministic given the inputs:
fixed bootstrap seed, sorted iteration, no timestamps.

Prints all tables to stdout and optionally writes a summary JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from h1_common import is_correct, normalize_answer

from chimera.cir.nodes import BetaDist

# Post-PR-9 the honest name is combine_pseudocount; pre-merge it is
# combine_ds. Same pseudocount-addition math either way.
_COMBINE = (
    BetaDist.combine_pseudocount
    if hasattr(BetaDist, "combine_pseudocount")
    else BetaDist.combine_ds
)

POOL_STRENGTH = 2.0  # per PREREG: weakest symmetric prior carrying the claim
ACCEPT_THRESHOLD = 0.80
N_BINS = 10
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261002


# ---------------------------------------------------------------- metrics

def brier(scores: list[float], labels: list[int]) -> float:
    n = len(scores)
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / n if n else float("nan")


def ece(scores: list[float], labels: list[int], n_bins: int = N_BINS) -> float:
    n = len(scores)
    if not n:
        return float("nan")
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [
            i for i, s in enumerate(scores)
            if (lo < s <= hi) or (b == 0 and s == 0.0)
        ]
        if not idx:
            continue
        acc = sum(labels[i] for i in idx) / len(idx)
        conf = sum(scores[i] for i in idx) / len(idx)
        total += abs(acc - conf) * len(idx) / n
    return total


def auroc(scores: list[float], labels: list[int]) -> float:
    n = len(scores)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(n), key=lambda i: scores[i])
    rank_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            if labels[order[k]] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def percentile_ci(values: list[float], low: float = 2.5, high: float = 97.5) -> tuple[float, float, int]:
    """Percentile CI over non-NaN values. Returns (lo, hi, n_dropped)."""
    clean = sorted(v for v in values if not math.isnan(v))
    dropped = len(values) - len(clean)
    if not clean:
        return float("nan"), float("nan"), dropped
    def pct(p: float) -> float:
        k = (len(clean) - 1) * p / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        return clean[f] if f == c else clean[f] + (clean[c] - clean[f]) * (k - f)
    return pct(low), pct(high), dropped


# ------------------------------------------------------- platt scaling

def fit_platt(xs: list[float], ys: list[int]) -> tuple[float, float]:
    """1-D logistic regression p = sigmoid(a*x + b) by Newton's method."""
    if not xs:
        return 0.0, 0.0
    if all(ys):
        return 0.0, 6.0
    if not any(ys):
        return 0.0, -6.0
    a, b = 0.0, 0.0
    for _ in range(50):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for x, y in zip(xs, ys):
            p = 1.0 / (1.0 + math.exp(-(a * x + b)))
            p = min(max(p, 1e-9), 1.0 - 1e-9)
            w = p * (1.0 - p)
            r = y - p
            g_a += r * x
            g_b += r
            h_aa += w * x * x
            h_ab += w * x
            h_bb += w
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (g_a * h_bb - g_b * h_ab) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a += da
        b += db
        if abs(da) < 1e-10 and abs(db) < 1e-10:
            break
    return a, b


def platt_apply(a: float, b: float, x: float) -> float:
    return 1.0 / (1.0 + math.exp(-(a * x + b)))


# ------------------------------------------------------------------ icc

def icc_one_way(matrix: list[list[float]]) -> float:
    """ICC(1,1): one-way random effects, balanced questions x samples."""
    n = len(matrix)
    if n < 2:
        return float("nan")
    k = len(matrix[0])
    if k < 2 or any(len(row) != k for row in matrix):
        return float("nan")
    grand = sum(sum(row) for row in matrix) / (n * k)
    ssb = sum((sum(row) / k - grand) ** 2 for row in matrix) * k
    ssw = sum((x - sum(row) / k) ** 2 for row in matrix for x in row)
    msb = ssb / (n - 1)
    msw = ssw / (n * (k - 1)) if n * (k - 1) else float("nan")
    denom = msb + (k - 1) * msw
    if denom == 0 or math.isnan(denom):
        return float("nan")
    return (msb - msw) / denom


def logit_clipped(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    v = math.log(p / (1.0 - p))
    return min(max(v, -6.0), 6.0)


# ------------------------------------------------------------------ io

def load_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_responses(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ------------------------------------------------------------------ arms

def majority_answer(samples: list[dict]) -> tuple[str, float]:
    """Return (majority answer, vote share). Ties -> earliest sample."""
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for i, s in enumerate(samples):
        key = normalize_answer(s["parsed_answer"])
        counts[key] += 1
        if key not in first_seen:
            first_seen[key] = i
    best = sorted(counts, key=lambda k: (-counts[k], first_seen[k]))[0]
    # recover the original (unnormalized) answer text of the earliest
    # sample carrying the winning normalized form
    for s in samples:
        if normalize_answer(s["parsed_answer"]) == best:
            return s["parsed_answer"], counts[best] / len(samples)
    raise AssertionError("unreachable")


def pool_confidences(confs: list[float]) -> tuple[float, bool]:
    """Shipped Beta pooling chain. Returns (pooled mean, conflicted).

    Each confidence becomes Beta(mean=c, strength=2.0) via
    BetaDist.from_confidence and they are combined in sample order with
    the shipped combine routine. On a conflict ValueError (K above
    threshold) falls back to the arithmetic mean and flags it.
    """
    dist = None
    for c in confs:
        b = BetaDist.from_confidence(max(1e-6, min(1.0 - 1e-6, c)), strength=POOL_STRENGTH)
        dist = b if dist is None else _COMBINE(dist, b)
    return dist.mean, False


def safe_pool(confs: list[float]) -> tuple[float, bool]:
    try:
        return pool_confidences(confs)
    except ValueError:
        return sum(confs) / len(confs), True


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze H1 pooling responses")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_N)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)
    gold = {row["id"]: row["gold_answers"] for row in dataset}
    order = [row["id"] for row in dataset]
    responses = load_responses(args.responses)

    modes = sorted({r["mode"] for r in responses})
    lines: list[str] = []
    summary: dict = {"dataset": args.dataset, "responses": args.responses,
                     "seed": args.seed, "bootstrap": args.bootstrap, "modes": {}}

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("== H1 pooling experiment ==")
    emit(f"dataset: {args.dataset} ({len(dataset)} questions) | "
         f"responses: {len(responses)} rows | seed: {args.seed} | "
         f"bootstrap resamples: {args.bootstrap}")
    emit("")

    n_calib = max(1, int(len(order) * 0.2))
    calib_ids = set(order[:n_calib])

    for mode in modes:
        by_q: dict[str, list[dict]] = {}
        for r in responses:
            if r["mode"] != mode:
                continue
            if not r.get("parse_ok") or r.get("parsed_confidence") is None:
                continue
            by_q.setdefault(r["id"], []).append(r)
        for qid in by_q:
            by_q[qid].sort(key=lambda s: (s["model"], s["sample_index"]))

        # inclusion: >=2 samples (B) / >=2 models (A)
        included: dict[str, list[dict]] = {}
        excluded = 0
        for qid, samples in by_q.items():
            ok = len(samples) >= 2 if mode == "B" else len({s["model"] for s in samples}) >= 2
            if ok:
                included[qid] = samples
            else:
                excluded += 1

        eval_ids = sorted(q for q in included if q not in calib_ids)
        calib_fit_ids = sorted(q for q in included if q in calib_ids)

        # per-question arm outputs: qid -> {arm: (p, y)}
        per_q: dict[str, dict] = {}
        pool_conflicts = 0
        for qid in sorted(included):
            samples = included[qid]
            confs = [s["parsed_confidence"] for s in samples]
            maj_ans, vote_share = majority_answer(samples)
            y_maj = 1 if is_correct(maj_ans, gold[qid]) else 0
            pooled_mean, conflicted = safe_pool(confs)
            pool_conflicts += 1 if conflicted else 0
            s0 = samples[0]
            y_s = 1 if is_correct(s0["parsed_answer"], gold[qid]) else 0
            per_q[qid] = {
                "S": (s0["parsed_confidence"], y_s),
                "M": (sum(confs) / len(confs), y_maj),
                "P": (pooled_mean, y_maj),
                "V": (vote_share, y_maj),
                "s_conf": s0["parsed_confidence"],
                "s_y": y_s,
            }

        # arm C: Platt scaling fit on calibration split S data
        cal_x = [per_q[q]["s_conf"] for q in calib_fit_ids]
        cal_y = [per_q[q]["s_y"] for q in calib_fit_ids]
        pa, pb = fit_platt(cal_x, cal_y)
        for qid in per_q:
            c, y = per_q[qid]["s_conf"], per_q[qid]["s_y"]
            per_q[qid]["C"] = (platt_apply(pa, pb, c), y)

        arms = ["S", "M", "P", "V", "C"]
        n_eval = len(eval_ids)
        metrics: dict[str, dict[str, float]] = {a: {} for a in arms}
        for a in arms:
            scores = [per_q[q][a][0] for q in eval_ids]
            labels = [per_q[q][a][1] for q in eval_ids]
            metrics[a]["brier"] = brier(scores, labels)
            metrics[a]["ece"] = ece(scores, labels)
            metrics[a]["auroc"] = auroc(scores, labels)

        # accepted set: P pooled mean >= threshold
        acc_ids = [q for q in eval_ids if per_q[q]["P"][0] >= ACCEPT_THRESHOLD]
        if acc_ids:
            acc_prec = sum(per_q[q]["P"][1] for q in acc_ids) / len(acc_ids)
            acc_claimed = sum(per_q[q]["P"][0] for q in acc_ids) / len(acc_ids)
        else:
            acc_prec, acc_claimed = float("nan"), float("nan")
        gap = acc_claimed - acc_prec

        # bootstrap
        rng = random.Random(args.seed)
        boot: dict[str, list[float]] = {f"{a}_{m}": [] for a in arms for m in ("brier", "ece", "auroc")}
        boot["gap"] = []
        boot["brier_diff_M_P"] = []
        auroc_dropped_total = 0
        for _ in range(args.bootstrap):
            picks = [eval_ids[rng.randrange(n_eval)] for _ in range(n_eval)] if n_eval else []
            for a in arms:
                sc = [per_q[q][a][0] for q in picks]
                lb = [per_q[q][a][1] for q in picks]
                boot[f"{a}_brier"].append(brier(sc, lb))
                boot[f"{a}_ece"].append(ece(sc, lb))
                boot[f"{a}_auroc"].append(auroc(sc, lb))
            acc = [q for q in picks if per_q[q]["P"][0] >= ACCEPT_THRESHOLD]
            if acc:
                p_ = sum(per_q[q]["P"][1] for q in acc) / len(acc)
                c_ = sum(per_q[q]["P"][0] for q in acc) / len(acc)
                boot["gap"].append(c_ - p_)
            else:
                boot["gap"].append(float("nan"))
            bm = brier([per_q[q]["M"][0] for q in picks], [per_q[q]["M"][1] for q in picks])
            bp = brier([per_q[q]["P"][0] for q in picks], [per_q[q]["P"][1] for q in picks])
            boot["brier_diff_M_P"].append(bm - bp)

        cis: dict[str, tuple[float, float, int]] = {}
        for k_, v in boot.items():
            cis[k_] = percentile_ci(v)

        # ICC(1,1) on logit(confidence): balanced subset
        counts = Counter(len(included[q]) for q in included)
        n0 = counts.most_common(1)[0][0] if counts else 0
        icc_qs = sorted(q for q in included if len(included[q]) == n0)
        icc_val = icc_one_way([[logit_clipped(s["parsed_confidence"]) for s in included[q]] for q in icc_qs])

        emit(f"[mode {mode}] questions included: {len(included)} "
             f"(eval {n_eval}, calib-fit {len(calib_fit_ids)}, excluded {excluded})")
        emit(f"arm   brier [95% CI]            ece [95% CI]              auroc [95% CI]")
        for a in arms:
            def fmt(pt: float, key: str) -> str:
                lo, hi, _ = cis[f"{a}_{key}"]
                return f"{pt:.4f} [{lo:.4f}, {hi:.4f}]"
            emit(f"{a:<5} {fmt(metrics[a]['brier'], 'brier'):24} "
                 f"{fmt(metrics[a]['ece'], 'ece'):24} {fmt(metrics[a]['auroc'], 'auroc')}")
        emit(f"accepted set (P pooled >= {ACCEPT_THRESHOLD}): n={len(acc_ids)}")
        glo, ghi, _ = cis["gap"]
        emit(f"  precision={acc_prec:.4f} mean_claimed={acc_claimed:.4f}")
        emit(f"[H1a] overconfidence gap (mode {mode}): {gap:.4f} [{glo:.4f}, {ghi:.4f}] "
             f"-> {'SUPPORTED' if mode == 'B' and gap > 0.05 and glo > 0 else 'not supported / n/a'}")
        dlo, dhi, _ = cis["brier_diff_M_P"]
        dpt = metrics["M"]["brier"] - metrics["P"]["brier"]
        verdict = "P beats M" if dlo > 0 else "adds nothing over averaging"
        emit(f"[H1b] Brier(M)-Brier(P): {dpt:.4f} [{dlo:.4f}, {dhi:.4f}] -> {verdict}")
        _, _, adrop = cis["S_auroc"]
        auroc_dropped_total = adrop
        emit(f"ICC(1,1) logit(conf): {icc_val:.4f} (n_questions={len(icc_qs)}, k={n0})")
        emit(f"pool conflicts (fallback to mean): {pool_conflicts} | "
             f"auroc bootstrap resamples dropped (single class): {auroc_dropped_total}")
        emit("")

        summary["modes"][mode] = {
            "n_included": len(included), "n_eval": n_eval,
            "n_calib_fit": len(calib_fit_ids), "n_excluded": excluded,
            "metrics": {a: {m: metrics[a][m] for m in ("brier", "ece", "auroc")} for a in arms},
            "cis": {k_: {"lo": v[0], "hi": v[1], "dropped": v[2]} for k_, v in cis.items()},
            "accepted": {"n": len(acc_ids), "precision": acc_prec, "mean_claimed": acc_claimed},
            "gap": gap, "brier_diff_M_P": dpt,
            "icc_logit": icc_val, "icc_k": n0, "icc_n": len(icc_qs),
            "pool_conflicts": pool_conflicts,
            "platt": {"a": pa, "b": pb},
        }

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        emit(f"summary written to {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
