# Decision Models as Research Instruments for Business Scholarship

Code and results for the paper *Decision Models as Research Instruments for Business Scholarship: Reliability and Valid Inference when Coding Corporate Climate Disclosures*.

Nine zero-shot coders (five decision-model configurations and four general-purpose language-model configurations) and the published ClimateBERT classifiers code seven climate-disclosure constructs. The study measures reliability against human labels (Krippendorff's alpha), the value of escalating uncertain passages to a reasoning model, the bias of prevalence estimates from machine codes, and the human coding saved by prediction-powered inference (PPI).

## Article
* **Journal**: Under Review
* **Title**: Decision Models as Research Instruments for Business Scholarship: Reliability and Valid Inference when Coding Corporate Climate Disclosures
* **DOI**: 

## Authors
* **Adj. Asst. Prof. Konstantinos I. Roumeliotis | University of Peloponnese**
* **Asoc. Prof. Dionisis Margaris | University of Peloponnese**
* **Asoc. Prof. Dimitris Spiliotopoulos | University of Peloponnese**
* **Prof. Nikolaos D. Tselikas | University of Peloponnese**

## Repository layout

```
bp08/                  experiment code (data preparation, prompts, runners, metrics, PPI, analysis)
  runners/             one runner per system family (hosted APIs, Clef, Perplexity Decider, vLLM LLMs, ClimateBERT)
scripts/               environment setup, model download, run orchestration, post hoc savings table
data/tasks/            manifest.json: item counts, label counts and SHA-256 of each constructed record file
results/main/raw/      every model prediction (one JSONL row per passage and system; no passage text)
results/main/tables/   analysis tables reported in the paper
results/main/tables_posthoc/   continuous human-coding savings, added after the run
```

## Data

Seven human-annotated datasets published with ClimateBERT, obtained from the Hugging Face Hub: `climatebert/climate_detection`, `climate_sentiment`, `climate_specificity`, `climate_commitments_actions`, `tcfd_recommendations`, `environmental_claims` and `netzero_reduction_data`. Most are licensed CC BY-NC-SA 4.0 and the net-zero data Apache-2.0. **The passages are not redistributed here.** `python -m bp08.data` downloads the datasets, pools the splits, removes duplicate texts and writes the request records to `data/records/`. For the net-zero data, which has no official split, a seeded random 30% becomes the test split. Compare the SHA-256 values that command writes to `data/tasks/manifest.json` with the committed file to check that you rebuilt the same records.

The raw predictions refer to passages by id (for example `climate_detection-00003`); the ids are assigned in the order of the rebuilt records.

## Systems

| System | Model / service |
|---|---|
| Jev 1.13 | TypeSafe AI hosted API, `jev-1.13.0` |
| Perplexity Decider | hosted API and local open weights, `pplx-decider-v1.1-27b` |
| Clef, Clef-flash | `Cloudflare/clef`, `Cloudflare/clef-flash` |
| Qwen3.8 | `Qwen/Qwen3.8-27B-FP8` as token readout, generative (JSON) and thinking |
| Gemma 4 | `google/gemma-4-31B-it`, generative |
| ClimateBERT | published supervised classifiers, used as a reference on the test split |

Stack: vLLM 0.31.0+cu129, torch 2.13.0+cu129, transformers 5.17.0 (main and Perplexity environments) and 5.10.2 (Clef). The study ran on 4× H100 NVL GPUs. Prompts and the codebook are in `bp08/data.py` and `bp08/prompts.py`.

## Reproducing the experiment

```bash
tcsh scripts/setup_server.csh          # creates .venv, .venv-clef and .venv-pplx
tcsh scripts/download_models.csh
export TYPESAFE_API_KEY=...            # Jev API
export PERPLEXITY_API_KEY=...          # Perplexity Decisions API

bash scripts/launch_tmux.sh prepare    # builds data/records
bash scripts/launch_tmux.sh full       # all systems -> results/main/raw
bash scripts/launch_tmux.sh analyze    # results/main/tables
python scripts/posthoc_savings.py      # results/main/tables_posthoc
```

`launch_tmux.sh full` starts one tmux session per GPU plus two sessions for the hosted APIs. Every runner can be resumed, and passages that are already finished are skipped.

To recompute the tables from the included predictions, run `python -m bp08.data` first (the analysis needs the gold labels in `data/records/`), then `python -m bp08.analyze --run-name main` and `python scripts/posthoc_savings.py`.

## Result tables

| File | Contents |
|---|---|
| `reliability.csv` | accuracy, macro-F1, Cohen's kappa, Krippendorff's alpha against the human labels (test split) |
| `cascade.csv`, `cascade_summary.csv` | escalation from each decision model to Qwen3.8 with thinking, by share escalated |
| `ppi.csv` | coverage, interval width, bias and RMSE of human-only, naive, PPI and PPI++ estimates by number of human-coded passages |
| `savings.csv` | human coding saved by PPI++ on the planned sample-size grid |
| `savings_continuous.csv` | human coding saved at equal interval width, for PPI and PPI++ (post hoc) |
| `efficiency.csv` | latency and cost |
| `coverage.csv` | answered and failed items per file |
