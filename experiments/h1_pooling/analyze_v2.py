"""V2 exploratory analysis for the H1 pooling experiment.

Reads ONLY responses.jsonl. No network. Fully deterministic given the
file: fixed seeds, sorted iteration, no timestamps. Stdlib only.

Implements PREREG_V2.md: arms K, P2, P10, S, M, V, Vc, VC
(mode A also V-loose); 5-fold cross-fitting for the calibrated arms;
Brier / ECE / AUROC / risk-coverage / AURC / guard simulation with
95 percent question-level bootstrap CIs; grading audit; falsifiable
expectation checks.
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
from analyze import (  # noqa: E402
    auroc,
    brier,
    derive_questions,
    ece,
    fit_platt,
    load_responses,
    majority_answer,
    percentile_ci,
)
from chimera.cir.nodes import BetaDist  # noqa: E402
from h1_common import is_correct, normalize_answer  # noqa: E402

CF_SEED = 20261002
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261002
AUDIT_SEED = 20261003
N_BINS = 10
GUARD_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]
COVERAGE_LEVELS = (20, 40, 60, 80, 100)


# ------------------------------------------------------------- grading

def token_f1(pred: str, gold_aliases: list[str]) -> float:
    """Max token-F1 of pred against any gold alias.

    Tokens are the normalized answer split on whitespace; F1 over
    token multiset overlap. 1.0 for identical, 0.0 for disjoint.
    """
    def toks(s: str) -> list[str]:
        return normalize_answer(s).split()
    pt = toks(pred)
    if not pt:
        return 0.0
    best = 0.0
    for g in gold_aliases:
        gt = toks(g)
        if not gt:
            continue
        common = Counter(pt) & Counter(gt)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        p = n_common / len(pt)
        r = n_common / len(gt)
        f1 = 2 * p * r / (p + r)
        best = max(best, f1)
    return best


# ------------------------------------------------------------- pooling

def pool_at_strength(confs: list[float], strength: float) -> tuple[float, bool]:
    """Shipped combine_pseudocount chain at a given strength.

    Returns (pooled mean, conflicted). On conflict ValueError falls
    back to the arithmetic mean, mirroring analyze.py.
    """
    dist = None
    for c in confs:
        b = BetaDist.from_confidence(max(1e-6, min(1.0 - 1e-6, c)), strength=strength)
        dist = b if dist is None else BetaDist.combine_pseudocount(dist, b)
    return dist.mean, False


def safe_pool_at(confs: list[float], strength: float) -> tuple[float, bool]:
    try:
        return pool_at_strength(confs, strength)
    except ValueError:
        return sum(confs) / len(confs), True


# ------------------------------------------------- logistic regression

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def fit_logistic_2d(xs: list[tuple[float, float]], ys: list[int]) -> tuple[float, float, float]:
    """2-D logistic regression p = sigmoid(a*x1 + b*x2 + c), Newton."""
    if not xs:
        return 0.0, 0.0, 0.0
    if all(ys):
        return 0.0, 0.0, 6.0
    if not any(ys):
        return 0.0, 0.0, -6.0
    a, b, c = 0.0, 0.0, 0.0
    for _ in range(50):
        g = [0.0, 0.0, 0.0]
        h = [[0.0] * 3 for _ in range(3)]
        for (x1, x2), y in zip(xs, ys):
            p = min(max(sigmoid(a * x1 + b * x2 + c), 1e-9), 1.0 - 1e-9)
            w = p * (1.0 - p)
            r = y - p
            f = (x1, x2, 1.0)
            for i in range(3):
                g[i] += r * f[i]
                for j in range(3):
                    h[i][j] += w * f[i] * f[j]
        # solve h * d = g via 3x3 inverse (Cramer)
        det = (
            h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
            - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
            + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0])
        )
        if abs(det) < 1e-12:
            break
        inv = _inv3(h, det)
        d = [sum(inv[i][j] * g[j] for j in range(3)) for i in range(3)]
        a, b, c = a + d[0], b + d[1], c + d[2]
        if max(abs(v) for v in d) < 1e-10:
            break
    return a, b, c


def _inv3(h, det):
    def m(r0, r1, c0, c1):
        return h[r0][c0] * h[r1][c1] - h[r0][c1] * h[r1][c0]
    return [
        [m(1, 2, 1, 2) / det, -m(0, 2, 1, 2) / det, m(0, 1, 1, 2) / det],
        [-m(1, 2, 0, 2) / det, m(0, 2, 0, 2) / det, -m(0, 1, 0, 2) / det],
        [m(1, 2, 0, 1) / det, -m(0, 2, 0, 1) / det, m(0, 1, 0, 1) / det],
    ]


# ------------------------------------------------------- cross-fitting

def k_fold(qids: list[str], k: int, seed: int) -> list[list[str]]:
    """Deterministic k-fold split of sorted qids via seeded shuffle."""
    rng = random.Random(seed)
    shuffled = sorted(qids)
    rng.shuffle(shuffled)
    folds = [[] for _ in range(k)]
    for i, q in enumerate(shuffled):
        folds[i % k].append(q)
    return [sorted(f) for f in folds]


def cross_fit_scores(qids, feats, labels, kind):
    """Out-of-fold scores for every qid. kind in {'K','Vc','VC'}.

    feats: qid -> feature (float for Vc, tuple for VC; unused for K).
    labels: qid -> 0/1.
    """
    folds = k_fold(qids, 5, CF_SEED)
    oof: dict[str, float] = {}
    for i, test in enumerate(folds):
        train = [q for j, f in enumerate(folds) if j != i for q in f]
        if kind == "K":
            base = sum(labels[q] for q in train) / len(train) if train else 0.0
            for q in test:
                oof[q] = base
        elif kind == "Vc":
            xs = [feats[q] for q in train]
            ys = [labels[q] for q in train]
            a, b = fit_platt(xs, ys)
            for q in test:
                oof[q] = sigmoid(a * feats[q] + b)
        elif kind == "VC":
            xs = [feats[q] for q in train]
            ys = [labels[q] for q in train]
            a, b, c = fit_logistic_2d(xs, ys)
            for q in test:
                x1, x2 = feats[q]
                oof[q] = sigmoid(a * x1 + b * x2 + c)
        else:
            raise ValueError(kind)
    return oof


# ------------------------------------------------------------- metrics

def risk_coverage(scores: list[float], labels: list[int],
                  levels: tuple = COVERAGE_LEVELS) -> dict[int, float]:
    """Accuracy among the top-coverage fraction by claimed score."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    out: dict[int, float] = {}
    for lv in levels:
        k = max(1, int(math.ceil(n * lv / 100.0)))
        top = order[:k]
        out[lv] = sum(labels[i] for i in top) / k
    return out


def aurc(scores: list[float], labels: list[int]) -> float:
    """Area under the risk-coverage curve (mean risk over coverages)."""
    n = len(scores)
    if not n:
        return float("nan")
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    correct = 0
    total = 0.0
    for k, i in enumerate(order, 1):
        correct += labels[i]
        total += 1.0 - correct / k
    return total / n


def guard_sim(scores: list[float], labels: list[int],
              thresholds: list[float] = GUARD_THRESHOLDS) -> dict[float, tuple[float, float]]:
    """Per threshold: (coverage, precision) of the accepted set."""
    n = len(scores)
    out: dict[float, tuple[float, float]] = {}
    for t in thresholds:
        acc = [i for i in range(n) if scores[i] >= t]
        if not acc:
            out[t] = (0.0, float("nan"))
        else:
            out[t] = (len(acc) / n, sum(labels[i] for i in acc) / len(acc))
    return out


def metric_bundle(scores: list[float], labels: list[int]) -> dict:
    return {
        "brier": brier(scores, labels),
        "ece": ece(scores, labels),
        "auroc": auroc(scores, labels),
        "aurc": aurc(scores, labels),
        "risk_cov": risk_coverage(scores, labels),
        "guard": guard_sim(scores, labels),
    }


# ------------------------------------------------------------------ main

ARMS_B = ["K", "P2", "P10", "S", "M", "V", "Vc", "VC"]
ARMS_A = ARMS_B + ["V-loose"]


def build_question_data(responses, gold, order, mode):
    """Per-question features for one mode. Returns (qids, feats, meta)."""
    by_q: dict[str, list[dict]] = {}
    for r in responses:
        if r["mode"] != mode:
            continue
        if not r.get("parse_ok") or r.get("parsed_confidence") is None:
            continue
        by_q.setdefault(r["id"], []).append(r)
    for qid in by_q:
        by_q[qid].sort(key=lambda s: (s["model"], s["sample_index"]))
    included = {}
    for qid, samples in by_q.items():
        ok = len(samples) >= 2 if mode == "B" else len({s["model"] for s in samples}) >= 2
        if ok:
            included[qid] = samples
    qids = sorted(included)
    feats: dict[str, dict] = {}
    for qid in qids:
        samples = included[qid]
        confs = [s["parsed_confidence"] for s in samples]
        maj_ans, vote_share = majority_answer(samples)
        y = 1 if is_correct(maj_ans, gold[qid]) else 0
        s0 = samples[0]
        y_s = 1 if is_correct(s0["parsed_answer"], gold[qid]) else 0
        p2, c2 = safe_pool_at(confs, 2.0)
        p10, c10 = safe_pool_at(confs, 10.0)
        vloose = None
        if mode == "A":
            agree = sum(1 for s in samples
                        if token_f1(s["parsed_answer"], [maj_ans]) >= 0.8)
            vloose = agree / len(samples)
        feats[qid] = {
            "confs": confs,
            "vote_share": vote_share,
            "mean_conf": sum(confs) / len(confs),
            "y": y,
            "s_conf": s0["parsed_confidence"],
            "s_y": y_s,
            "s_ans": s0["parsed_answer"],
            "maj_ans": maj_ans,
            "P2": p2, "P10": p10,
            "conflicts": (c2, c10),
            "V-loose": vloose,
            "qidx": samples[0]["question_index"],
        }
    return qids, feats


def arm_scores(arm, qids, feats, oof):
    """Raw or out-of-fold scores+labels for an arm over qids."""
    scores, labels = [], []
    for q in qids:
        f = feats[q]
        if arm == "S":
            scores.append(f["s_conf"]); labels.append(f["s_y"])
        elif arm == "M":
            scores.append(f["mean_conf"]); labels.append(f["y"])
        elif arm == "V":
            scores.append(f["vote_share"]); labels.append(f["y"])
        elif arm == "V-loose":
            scores.append(f["V-loose"]); labels.append(f["y"])
        elif arm in ("P2", "P10", "K", "Vc", "VC"):
            scores.append(oof[arm][q]); labels.append(f["y"])
        else:
            raise ValueError(arm)
    return scores, labels


def main() -> int:
    ap = argparse.ArgumentParser(description="V2 exploratory H1 analysis")
    ap.add_argument("--responses", required=True)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_N)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    responses = load_responses(args.responses)
    gold, order, ds_names, ds_shas = derive_questions(responses)

    lines: list[str] = []
    def emit(s=""):
        lines.append(s)
        print(s)

    emit("== H1 v2 exploratory analysis ==")
    emit(f"responses: {len(responses)} rows | dataset: {ds_names}")

    results: dict = {"modes": {}}
    audit_items: list[dict] = []

    for mode in ("B", "A"):
        arms = ARMS_B if mode == "B" else ARMS_A
        qids, feats = build_question_data(responses, gold, order, mode)
        if not qids:
            emit("")
            emit(f"[mode {mode}] n=0 questions, skipped")
            results["modes"][mode] = {"n": 0, "schemes": {}, "expectations": {},
                                      "conflicts": {}}
            continue
        order_pos = {q: i for i, q in enumerate(order) if q in feats}
        labels_all = {q: feats[q]["y"] for q in qids}

        # out-of-fold scores for calibrated arms (5-fold cross-fitting)
        oof: dict[str, dict[str, float]] = {}
        oof["K"] = cross_fit_scores(
            qids, {}, labels_all, "K")
        oof["Vc"] = cross_fit_scores(
            qids, {q: feats[q]["vote_share"] for q in qids}, labels_all, "Vc")
        oof["VC"] = cross_fit_scores(
            qids, {q: tuple([feats[q]["vote_share"], feats[q]["mean_conf"]]) for q in qids},
            labels_all, "VC")
        # raw pooled arms
        oof["P2"] = {q: feats[q]["P2"] for q in qids}
        oof["P10"] = {q: feats[q]["P10"] for q in qids}
        n_conflicts = {
            "P2": sum(1 for q in qids if feats[q]["conflicts"][0]),
            "P10": sum(1 for q in qids if feats[q]["conflicts"][1]),
        }

        # 20/80 split for comparability: first 100 by dataset order
        calib_ids = set(sorted(qids, key=lambda q: feats[q]["qidx"])[:100])
        eval80 = sorted([q for q in qids if q not in calib_ids],
                        key=lambda q: feats[q]["qidx"])
        oof80: dict[str, dict[str, float]] = {}
        tr = [q for q in qids if q in calib_ids]
        base80 = sum(labels_all[q] for q in tr) / len(tr)
        oof80["K"] = {q: base80 for q in eval80}
        xs = [feats[q]["vote_share"] for q in tr]
        ys = [labels_all[q] for q in tr]
        a1, b1 = fit_platt(xs, ys)
        oof80["Vc"] = {q: sigmoid(a1 * feats[q]["vote_share"] + b1) for q in eval80}
        xs2 = [(feats[q]["vote_share"], feats[q]["mean_conf"]) for q in tr]
        a2, b2, c2 = fit_logistic_2d(xs2, ys)
        oof80["VC"] = {q: sigmoid(a2 * feats[q]["vote_share"] + b2 * feats[q]["mean_conf"] + c2)
                       for q in eval80}
        oof80["P2"] = {q: feats[q]["P2"] for q in eval80}
        oof80["P10"] = {q: feats[q]["P10"] for q in eval80}

        mode_res: dict = {"n": len(qids), "schemes": {}}
        for scheme, sqids, soof in (("5fold", qids, oof), ("split2080", eval80, oof80)):
            bundles, boots = {}, {}
            for arm in arms:
                sc, lb = arm_scores(arm, sqids, feats, soof)
                bundles[arm] = metric_bundle(sc, lb)
            # bootstrap: shared resamples per scheme
            rng = random.Random(args.seed + (0 if scheme == "5fold" else 1))
            n = len(sqids)
            draws = [[rng.randrange(n) for _ in range(n)] for _ in range(args.bootstrap)]
            for arm in arms:
                sc, lb = arm_scores(arm, sqids, feats, soof)
                bb = {"brier": [], "ece": [], "auroc": [], "aurc": [],
                      "risk_cov": {lv: [] for lv in COVERAGE_LEVELS},
                      "guard_cov": {t: [] for t in GUARD_THRESHOLDS},
                      "guard_prec": {t: [] for t in GUARD_THRESHOLDS}}
                for d in draws:
                    rsc = [sc[i] for i in d]
                    rlb = [lb[i] for i in d]
                    bb["brier"].append(brier(rsc, rlb))
                    bb["ece"].append(ece(rsc, rlb))
                    bb["auroc"].append(auroc(rsc, rlb))
                    bb["aurc"].append(aurc(rsc, rlb))
                    rc = risk_coverage(rsc, rlb)
                    for lv in COVERAGE_LEVELS:
                        bb["risk_cov"][lv].append(rc[lv])
                    gs = guard_sim(rsc, rlb)
                    for t in GUARD_THRESHOLDS:
                        cov, prec = gs[t]
                        bb["guard_cov"][t].append(cov)
                        bb["guard_prec"][t].append(prec)
                cis = {}
                for k_ in ("brier", "ece", "auroc", "aurc"):
                    lo, hi, dr = percentile_ci(bb[k_])
                    cis[k_] = {"lo": lo, "hi": hi, "dropped": dr}
                cis["risk_cov"] = {lv: {"lo": v[0], "hi": v[1]}
                                   for lv, v in
                                   ((lv_, percentile_ci(bb["risk_cov"][lv_])[:2])
                                    for lv_ in COVERAGE_LEVELS)}
                cis["guard"] = {t: {"cov": percentile_ci(bb["guard_cov"][t])[:2],
                                    "prec": percentile_ci(bb["guard_prec"][t])[:2]}
                                for t in GUARD_THRESHOLDS}
                boots[arm] = cis
            mode_res["schemes"][scheme] = {"n": n, "bundles": bundles, "cis": boots}

        # expectation checks (5-fold scheme)
        b5 = mode_res["schemes"]["5fold"]["bundles"]
        c5 = mode_res["schemes"]["5fold"]["cis"]
        exp: dict[str, dict] = {}
        # E1: P10 saturation for confidences >= 0.9
        sat_q = [q for q in qids if all(c >= 0.9 for c in feats[q]["confs"])]
        sat_vals = sorted(oof["P10"][q] for q in sat_q)
        med = sat_vals[len(sat_vals) // 2] if sat_vals else float("nan")
        exp["E1_P10_saturation"] = {
            "n": len(sat_q), "median_P10": med, "met": bool(sat_vals) and med >= 0.99,
        }
        # E2: mode A Vc, VC beat K on Brier
        if mode == "A":
            exp["E2_cal_beats_K"] = {
                "brier_K": b5["K"]["brier"], "brier_Vc": b5["Vc"]["brier"],
                "brier_VC": b5["VC"]["brier"],
                "met": b5["Vc"]["brier"] < b5["K"]["brier"] and b5["VC"]["brier"] < b5["K"]["brier"],
            }
        # E3: mode B no verbalized-confidence arm AUROC > 0.6
        if mode == "B":
            aucs = {a: b5[a]["auroc"] for a in ("S", "M", "P2", "P10")}
            exp["E3_no_auc_above_06"] = {
                "aurocs": {a: round(v, 4) for a, v in aucs.items()},
                "met": all(v <= 0.6 for v in aucs.values()),
            }
        mode_res["expectations"] = exp
        mode_res["conflicts"] = n_conflicts
        results["modes"][mode] = mode_res

        # ---- print ----
        emit("")
        emit(f"[mode {mode}] n={len(qids)} questions")
        for scheme in ("5fold", "split2080"):
            scm = mode_res["schemes"][scheme]
            emit(f"  -- scheme {scheme} (n={scm['n']}) --")
            emit("  arm     brier [95% CI]            ece [95% CI]              auroc [95% CI]            aurc [95% CI]")
            for arm in arms:
                bnd, ci_ = scm["bundles"][arm], scm["cis"][arm]
                def fmt(pt, k_):
                    lo, hi = ci_[k_]["lo"], ci_[k_]["hi"]
                    return f"{pt:.4f} [{lo:.4f},{hi:.4f}]"
                emit(f"  {arm:<8}{fmt(bnd['brier'],'brier'):26}{fmt(bnd['ece'],'ece'):26}"
                     f"{fmt(bnd['auroc'],'auroc'):26}{fmt(bnd['aurc'],'aurc')}")
            emit("  risk-coverage accuracy:")
            for arm in arms:
                rc = scm["bundles"][arm]["risk_cov"]
                rcc = scm["cis"][arm]["risk_cov"]
                cells = " ".join(f"{lv}:{rc[lv]:.3f}[{rcc[lv]['lo']:.3f},{rcc[lv]['hi']:.3f}]"
                                 for lv in COVERAGE_LEVELS)
                emit(f"    {arm:<8}{cells}")
            emit("  guard simulation: coverage / precision at thresholds")
            for arm in arms:
                gs = scm["bundles"][arm]["guard"]
                cells = " ".join(f"{t:.2f}:{gs[t][0]:.2f}/{gs[t][1]:.3f}"
                                 if not math.isnan(gs[t][1]) else f"{t:.2f}:{gs[t][0]:.2f}/nan"
                                 for t in GUARD_THRESHOLDS)
                emit(f"    {arm:<8}{cells}")
        emit(f"  pool conflicts (fallback to mean): P2={n_conflicts['P2']} P10={n_conflicts['P10']}")
        for name, e in exp.items():
            emit(f"  EXPECTATION {name}: {'MET' if e['met'] else 'NOT MET'} "
                 f"{json.dumps({k: v for k, v in e.items() if k != 'met'}, sort_keys=True)}")

    # ---- grading audit ----
    emit("")
    emit("== grading audit (40 seeded items) ==")
    b_rows = [r for r in responses if r["mode"] == "B" and r.get("parse_ok")]
    b_qids = sorted({r["id"] for r in b_rows})
    rng = random.Random(AUDIT_SEED)
    audit = rng.sample(b_qids, min(40, len(b_qids)))
    for q in audit:
        rows = [r for r in b_rows if r["id"] == q]
        rows.sort(key=lambda s: s["sample_index"])
        gold_a = rows[0]["gold_answers"]
        maj, _ = majority_answer(rows)
        y = is_correct(maj, gold_a)
        f1 = token_f1(maj, gold_a)
        audit_items.append({"id": q, "gold": gold_a, "pred": maj,
                            "correct_exact": bool(y), "token_f1": round(f1, 3)})
        emit(f"  {q} gold={gold_a} pred={maj!r} exact={bool(y)} f1={f1:.3f}")
    # looser-grader sensitivity per arm/mode
    emit("  looser-grader accuracy (token-F1 >= 0.8):")
    for mode in ("B", "A"):
        arms = ARMS_B if mode == "B" else ARMS_A
        qids, feats = build_question_data(responses, gold, order, mode)
        if not qids:
            emit(f"    mode {mode}: no questions, skipped")
            continue
        line = []
        for arm in arms:
            ok = 0
            for q in qids:
                f = feats[q]
                pred = f["s_ans"] if arm == "S" else f["maj_ans"]
                if token_f1(pred, gold[q]) >= 0.8:
                    ok += 1
            line.append(f"{arm}={ok/len(qids):.4f}")
        emit(f"    mode {mode}: " + " ".join(line))
    results["audit"] = audit_items

    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        emit(f"v2 output written to {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
