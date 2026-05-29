# Evaluation CLI Flags

Run from the project root:

```powershell
.\evaluate.cmd --pattern "engineering*.txt" --skip-judge --require-expected
```

The wrappers use `.\.venv\Scripts\python.exe` when it exists, otherwise they fall back to `python`.
Use `evaluate.cmd` on Windows if PowerShell script execution is disabled. `evaluate.ps1` is also available when scripts are allowed.

## Flags

| Flag | Default | What it does |
| --- | --- | --- |
| `--dataset-dir PATH` | `app/api/evaluation/dataset` | Directory with transcript `.txt` files. |
| `--expected-dir PATH` | `app/api/evaluation/expected` | Directory with `*.expected.json` ground-truth files. |
| `--output-dir PATH` | `app/api/evaluation/reports` | Directory where `summary.json` and detailed reports are written. |
| `--pattern GLOB` | `*.txt` | Selects which transcripts to run, for example `"engineering*.txt"`. |
| `--limit N` | none | Runs only the first `N` matched transcripts. |
| `--provider gigachat\|deepseek` | `.env` `LLM_PROVIDER` | Chooses the extraction provider. |
| `--extractor-model MODEL` | provider env default | Temporarily overrides the extraction model for this evaluation run. |
| `--model MODEL` | provider env default | Deprecated alias for `--extractor-model`. |
| `--skip-judge` | off | Skips the LLM judge and runs deterministic scoring only. |
| `--judge-provider gigachat\|deepseek` | extractor provider | Chooses the LLM judge provider independently from extraction. |
| `--judge-model MODEL` | judge provider env default | Temporarily overrides the judge model for this evaluation run. |
| `--skip-deterministic` | off | Skips expected JSON comparison and runs extraction/judge only. |
| `--require-expected` | off | Treats a missing expected JSON as an error for that transcript. |
| `--fail-under SCORE` | none | Exits with failure if average deterministic score is below `SCORE`. |
| `--no-transcript-in-report` | off | Omits full transcript text from detailed report JSONs. |
| `--fail-fast` | off | Stops on the first transcript error instead of continuing. |

## Common Commands

Run deterministic scoring against the four engineering expected files:

```powershell
.\evaluate.cmd --pattern "engineering*.txt" --skip-judge --require-expected
```

Run with DeepSeek and fail if average deterministic score is below `0.75`:

```powershell
.\evaluate.cmd --provider deepseek --pattern "engineering*.txt" --skip-judge --require-expected --fail-under 0.75
```

Run extraction with DeepSeek and judge with GigaChat:

```powershell
.\evaluate.cmd --provider deepseek --judge-provider gigachat --pattern "engineering*.txt"
```

Run extraction and judge with explicit models:

```powershell
.\evaluate.cmd --provider deepseek --extractor-model deepseek-chat --judge-provider deepseek --judge-model deepseek-chat --pattern "engineering*.txt"
```

Run one transcript and keep judge enabled:

```powershell
.\evaluate.cmd --pattern "engineering1.txt" --limit 1
```

Write reports to a separate directory:

```powershell
.\evaluate.cmd --pattern "engineering*.txt" --output-dir app/api/evaluation/reports_engineering
```
