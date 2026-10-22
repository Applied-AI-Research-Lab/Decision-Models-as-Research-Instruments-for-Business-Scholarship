from __future__ import annotations

from ..config import CLIMATEBERT
from ..utils import Timer, append_jsonl, get_logger
from .common import base_parser, jobs, pred_row

log = get_logger("climatebert")
MODEL_KEY = "climatebert_ref"
ALIASES = {"non": "non-specific", "spec": "specific"}


def label_map(id2label: dict, options: list[str]) -> dict[int, str]:
    out = {}
    for i, lab in id2label.items():
        name = ALIASES.get(str(lab).lower(), str(lab).lower())
        if name not in options:
            raise ValueError(f"model label {lab!r} has no matching option in {options}")
        out[int(i)] = name
    return out


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    args = base_parser("Run the published ClimateBERT classifiers").parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for task, cond, recs, out in jobs(args, MODEL_KEY):
        recs = [r for r in recs if r["meta"]["split"] == "test"]
        if not recs:
            continue
        tok = AutoTokenizer.from_pretrained(CLIMATEBERT[task])
        model = AutoModelForSequenceClassification.from_pretrained(CLIMATEBERT[task]).to(device).eval()
        lmap = label_map(model.config.id2label, recs[0]["options"])
        rows = []
        for i in range(0, len(recs), 32):
            chunk = recs[i:i + 32]
            enc = tok([r["text"] for r in chunk], truncation=True, max_length=512, padding=True, return_tensors="pt").to(device)
            with Timer() as t, torch.inference_mode():
                P = model(**enc).logits.float().softmax(-1).cpu().tolist()
            for r, p in zip(chunk, P):
                probs = {o: 0.0 for o in r["options"]}
                for k, v in enumerate(p):
                    probs[lmap[k]] = v
                rows.append(pred_row(r, MODEL_KEY, probs, t.seconds / len(chunk)))
        append_jsonl(out, rows)
        log.info("%s: %d test items, label map %s", task, len(rows), lmap)


if __name__ == "__main__":
    main()
