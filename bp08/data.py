from __future__ import annotations

import argparse
import hashlib
import json

from .config import NETZERO_TEST, SEED, TASKS, TASKS_DIR, records_path
from .utils import get_logger, write_jsonl

log = get_logger("data")

CODEBOOK = {
    "climate_detection": {
        "repo": "climatebert/climate_detection", "names": ["no", "yes"], "target": "yes",
        "instructions": "Is this paragraph from a corporate report about climate-related topics?",
        "criteria": {"yes": "Yes: it discusses climate change, greenhouse-gas emissions, the energy transition, or climate-related risks or opportunities.",
                     "no": "No: it is not about climate-related topics."}},
    "climate_sentiment": {
        "repo": "climatebert/climate_sentiment", "names": ["risk", "neutral", "opportunity"], "target": "risk",
        "instructions": "How does this climate-related paragraph frame climate change for the company?",
        "criteria": {"risk": "Risk: climate change is presented as a threat, cost or risk for the company.",
                     "neutral": "Neutral: no clear risk or opportunity framing.",
                     "opportunity": "Opportunity: climate change is presented as an opportunity, e.g. new markets or products."}},
    "climate_specificity": {
        "repo": "climatebert/climate_specificity", "names": ["non-specific", "specific"], "target": "specific",
        "instructions": "Is this climate-related paragraph specific or non-specific?",
        "criteria": {"specific": "Specific: concrete, verifiable statements such as quantified targets, dates, named measures or investments.",
                     "non-specific": "Non-specific: vague or generic statements without concrete, verifiable detail."}},
    "climate_commitments_actions": {
        "repo": "climatebert/climate_commitments_actions", "names": ["no", "yes"], "target": "yes",
        "instructions": "Does this climate-related paragraph describe the company's climate commitments or actions?",
        "criteria": {"yes": "Yes: it states targets, commitments, measures or actions the company takes on climate change.",
                     "no": "No: it does not describe the company's own commitments or actions."}},
    "tcfd_recommendations": {
        "repo": "climatebert/tcfd_recommendations", "names": ["none", "metrics", "strategy", "risk", "governance"],
        "target": "metrics",
        "instructions": "Which TCFD disclosure category does this climate-related paragraph belong to?",
        "criteria": {"governance": "Governance: board oversight and management's role on climate-related issues.",
                     "strategy": "Strategy: climate-related risks and opportunities and their impact on business, strategy and planning.",
                     "risk": "Risk management: processes for identifying, assessing and managing climate-related risks.",
                     "metrics": "Metrics and targets: metrics, emissions figures and targets used to manage climate issues.",
                     "none": "None: it does not fit any TCFD category."}},
    "environmental_claims": {
        "repo": "climatebert/environmental_claims", "names": ["no", "yes"], "target": "yes",
        "instructions": "Is this sentence an environmental claim?",
        "criteria": {"yes": "Yes: the company claims that it, its products or its activities are environmentally beneficial or less harmful.",
                     "no": "No: it is not an environmental claim."}},
    "netzero_reduction": {
        "repo": "climatebert/netzero_reduction_data", "names": None, "target": "net-zero",
        "instructions": "Does this text state an emissions target, and if so which kind?",
        "criteria": {"net-zero": "Net-zero: a target of net-zero emissions or carbon neutrality.",
                     "reduction": "Reduction: an emissions reduction target that is not a net-zero target.",
                     "none": "None: no emissions target."}},
}


def load_task(task: str, limit=None):
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files
    cb = CODEBOOK[task]
    if task == "netzero_reduction":
        df = pd.read_csv(hf_hub_download(cb["repo"], "targets_final.csv", repo_type="dataset"))
        df = df.rename(columns={"target": "label"})[["text", "label"]].dropna()
        test_idx = df.sample(frac=0.30, random_state=SEED).index
        df["split"] = ["test" if i in set(test_idx) else "train" for i in df.index]
    else:
        parts = []
        for f in list_repo_files(cb["repo"], repo_type="dataset"):
            if f.startswith("data/") and f.endswith(".parquet"):
                split = "test" if "/test-" in f else "train"
                d = pd.read_parquet(hf_hub_download(cb["repo"], f, repo_type="dataset"))
                parts.append(d.assign(label=d.label.map(lambda i: cb["names"][int(i)]), split=split)[["text", "label", "split"]])
        df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates("text").reset_index(drop=True)
    if limit:
        df = df.groupby(["split", "label"], group_keys=False).head(max(1, limit // (2 * df.label.nunique())))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    manifest = []
    for task in TASKS:
        cb = CODEBOOK[task]
        df = load_task(task, args.limit)
        opts = list(cb["criteria"])
        rows = [{"id": f"{task}-{i:05d}", "task": task, "condition": "canonical", "text": t,
                 "state": f"Passage from a corporate report: {t}", "gold": l, "instructions": cb["instructions"],
                 "options": opts, "criteria": dict(cb["criteria"]), "meta": {"target": cb["target"], "split": sp}}
                for i, (t, l, sp) in enumerate(zip(df.text, df.label, df.split))]
        assert all(r["gold"] in opts for r in rows), task
        write_jsonl(records_path(task, "canonical"), rows)
        manifest.append({"task": task, "n": len(rows), "n_test": int((df.split == "test").sum()),
                         "labels": df.label.value_counts().to_dict(),
                         "sha256": hashlib.sha256(records_path(task, "canonical").read_bytes()).hexdigest()})
        log.info("%s: %d items (test %d) %s", task, len(rows), int((df.split == "test").sum()), df.label.value_counts().to_dict())
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    (TASKS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
