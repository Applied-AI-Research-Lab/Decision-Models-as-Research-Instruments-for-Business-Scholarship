import sys
import pandas as pd
from bp08.config import results_dir

run = sys.argv[1] if len(sys.argv) > 1 else "main"
p = pd.read_csv(results_dir(run) / "tables" / "ppi.csv")
h = p[p.estimator == "human_only"].set_index(["model", "task", "n_human"]).ci_width.rename("w_human")
rows = []
for est in ("ppi", "ppi++"):
    e = p[p.estimator == est].set_index(["model", "task", "n_human"])
    d = e[["ci_width", "coverage"]].join(h)
    d["human_coding_saved_share"] = 1 - (d.ci_width / d.w_human) ** 2
    d["estimator"] = est
    rows.append(d.reset_index())
out = pd.concat(rows)
dst = results_dir(run) / "tables_posthoc"
dst.mkdir(parents=True, exist_ok=True)
out.to_csv(dst / "savings_continuous.csv", index=False)
print(out.groupby(["estimator", "n_human"])[["human_coding_saved_share", "coverage"]].agg(["mean", "min", "max"]).round(3).to_string())
