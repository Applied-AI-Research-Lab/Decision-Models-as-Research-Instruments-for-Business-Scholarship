from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import metrics as M
from .config import API_PROVIDERS, GPU_HOUR_USD, HUMAN_SAMPLE_SIZES, PPI_REPS, SEED, TASKS, records_path, results_dir
from .ppi import krippendorff_alpha_nominal, simulate
from .utils import get_logger, iter_jsonl, latest_ok, read_jsonl

log = get_logger("analyze")
DECISION = ["jev", "pplx_api", "clef", "clef_flash", "pplx"]
SECOND = "qwen_think"


def load(run, model, task):
    rp, pp = records_path(task, "canonical"), results_dir(run) / "raw" / model / task / "canonical.jsonl"
    if not rp.exists() or not pp.exists():
        return None
    recs, preds = read_jsonl(rp), latest_ok(pp)
    n_err = sum(1 for r in iter_jsonl(pp) if r.get("error") and r["id"] not in preds)
    recs = [r for r in recs if r["id"] in preds]
    if not recs:
        return None
    opts = recs[0]["options"]
    P = np.array([[preds[r["id"]]["probs"].get(o, 0.0) for o in opts] for r in recs], float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    rows = list(preds.values())
    cost = (float(np.mean([r["input_tokens"] or 0 for r in rows])) * API_PROVIDERS[model]["usd_per_m_input"] / 1e6
            if model in API_PROVIDERS else float(np.mean([r["latency_s"] or 0 for r in rows])) * GPU_HOUR_USD / 3600)
    return {"ids": [r["id"] for r in recs], "P": P, "y": np.array([opts.index(r["gold"]) for r in recs]), "opts": opts,
            "split": np.array([r["meta"]["split"] for r in recs]), "target": recs[0]["meta"]["target"],
            "pos": {r["id"]: k for k, r in enumerate(recs)}, "n_err": n_err, "cost": cost}


def reliability(cache):
    rows = []
    from sklearn.metrics import cohen_kappa_score
    for (m, t), d in cache.items():
        k = d["split"] == "test"
        if k.sum() < 10:
            continue
        P, y = d["P"][k], d["y"][k]
        pred = P.argmax(1)
        rows.append({"model": m, "task": t, "n_test": int(k.sum()), "accuracy": M.accuracy(P, y),
                     "macro_f1": M.macro_f1(P, y), "cohen_kappa": float(cohen_kappa_score(y, pred)),
                     "krippendorff_alpha": krippendorff_alpha_nominal(y, pred), "ece": M.ece_equal_mass(P, y)})
    return rows


def cascade(cache):
    rows, summary = [], []
    for t in TASKS:
        L = cache.get((SECOND, t))
        if L is None:
            continue
        for m in DECISION:
            d = cache.get((m, t))
            if d is None:
                continue
            ids = [i for i in d["ids"] if i in L["pos"] and d["split"][d["pos"][i]] == "test"]
            if len(ids) < 20:
                continue
            Pd, Pl = d["P"][[d["pos"][i] for i in ids]], L["P"][[L["pos"][i] for i in ids]]
            y = d["y"][[d["pos"][i] for i in ids]]
            conf = Pd.max(1)
            alpha_llm = krippendorff_alpha_nominal(y, Pl.argmax(1))
            best = None
            for x in np.round(np.arange(0, 1.0001, 0.05), 2):
                thr = np.quantile(conf, x) if 0 < x < 1 else (-np.inf if x == 0 else np.inf)
                esc = conf < thr
                pred = np.where(esc, Pl.argmax(1), Pd.argmax(1))
                a = krippendorff_alpha_nominal(y, pred)
                cost = d["cost"] + esc.mean() * L["cost"]
                rows.append({"model": m, "task": t, "escalation_share": x, "krippendorff_alpha": a,
                             "cost_per_1k": cost * 1000, "cost_vs_llm_only": cost / L["cost"] if L["cost"] else np.nan})
                if best is None and a >= alpha_llm - 0.01:
                    best = (x, cost / L["cost"] if L["cost"] else np.nan)
            summary.append({"model": m, "task": t, "alpha_decision": krippendorff_alpha_nominal(y, Pd.argmax(1)),
                            "alpha_llm": alpha_llm, "min_escalation_to_match_llm": best[0] if best else np.nan,
                            "cost_vs_llm_only_at_match": best[1] if best else np.nan})
    return rows, summary


def ppi_tables(cache, reps, sizes=tuple(HUMAN_SAMPLE_SIZES)):
    rows, savings = [], []
    rng = np.random.default_rng(SEED)
    for (m, t), d in cache.items():
        if m == "climatebert_ref":
            continue
        k = d["opts"].index(d["target"])
        Y = (d["y"] == k).astype(float)
        f = d["P"][:, k]
        fh = (d["P"].argmax(1) == k).astype(float)
        N = len(Y)
        widths = {}
        for n in sizes:
            if n > N // 2:
                continue
            res = simulate(Y, f, fh, n, reps, rng)
            for est, v in res.items():
                rows.append({"model": m, "task": t, "N": N, "n_human": n, "estimator": est, "truth": Y.mean(), **v})
                widths[(est, n)] = v["ci_width"]
        ref_n = max([n for n in sizes if n <= N // 2], default=None)
        if ref_n and ("human_only", ref_n) in widths:
            target = widths[("human_only", ref_n)]
            need = next((n for n in sizes if widths.get(("ppi++", n), np.inf) <= target), None)
            savings.append({"model": m, "task": t, "human_only_n": ref_n, "human_only_width": target,
                            "ppi++_n_for_same_width": need,
                            "human_coding_saved_share": (1 - need / ref_n) if need else 0.0})
    return rows, savings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--reps", type=int, default=PPI_REPS)
    ap.add_argument("--sizes", default=",".join(map(str, HUMAN_SAMPLE_SIZES)), help="human sample sizes")
    args = ap.parse_args()
    out_t, out_f = results_dir(args.run_name) / "tables", results_dir(args.run_name) / "figures"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    raw = results_dir(args.run_name) / "raw"
    models = sorted(p.name for p in raw.iterdir() if p.is_dir()) if raw.exists() else []
    cache, cov, eff = {}, [], []
    for m in models:
        for t in TASKS:
            d = load(args.run_name, m, t)
            if d is not None:
                cache[(m, t)] = d
                cov.append({"model": m, "task": t, "answered": len(d["ids"]), "unresolved_errors": d["n_err"]})
                eff.append({"model": m, "task": t, "usd_per_10k_passages": d["cost"] * 10000})
    casc, casc_sum = cascade(cache)
    ppi_rows, savings = ppi_tables(cache, args.reps, tuple(int(x) for x in args.sizes.split(",")))
    tables = {"coverage": cov, "efficiency": eff, "reliability": reliability(cache), "cascade": casc,
              "cascade_summary": casc_sum, "ppi": ppi_rows, "savings": savings}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out_t / f"{name}.csv", index=False)
    log.info("tables written to %s", out_t)
    make_figures(pd.DataFrame(tables["reliability"]), pd.DataFrame(ppi_rows), out_f)


def make_figures(rel, ppi, out_f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(rel):
        piv = rel.pivot_table(index="task", columns="model", values="krippendorff_alpha")
        fig, ax = plt.subplots(figsize=(1.2 * len(piv.columns) + 4, 0.5 * len(piv) + 2))
        im = ax.imshow(piv.values, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=7)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                if np.isfinite(piv.values[i, j]):
                    ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w")
        fig.colorbar(im, label="Krippendorff's alpha"); fig.tight_layout(); fig.savefig(out_f / "reliability.png", dpi=200); plt.close(fig)
    if len(ppi):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        g = ppi.groupby(["estimator", "n_human"])[["coverage", "ci_width"]].mean().reset_index()
        for est, gg in g.groupby("estimator"):
            axes[0].plot(gg.n_human, gg.coverage, marker="o", label=est)
            axes[1].plot(gg.n_human, gg.ci_width, marker="o", label=est)
        axes[0].axhline(0.95, color="k", ls="--", lw=0.8); axes[0].set_title("coverage (mean over constructs and models)")
        axes[1].set_title("95% CI width"); axes[1].legend(fontsize=7)
        for ax in axes:
            ax.set_xlabel("human-coded passages (n)")
        fig.tight_layout(); fig.savefig(out_f / "ppi.png", dpi=200); plt.close(fig)
    log.info("figures written to %s", out_f)


if __name__ == "__main__":
    main()
