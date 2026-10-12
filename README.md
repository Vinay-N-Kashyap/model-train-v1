# Small Language Model Training

This is the isolated research workspace for the from-scratch interview-answer model described by `Small_Language_Model_SRS_Baseline_v1.0.pdf`. It does not change or depend on the LMS application code.

## Current status

- A byte-level decoder-only Transformer and local-corpus training loop are scaffolded.
- The default configuration is about 4.9 million parameters; it is a pipeline prototype, not an interview evaluator.
- Training has not started. No dataset has been approved or downloaded.
- Current machine: 16 GB RAM, 12th Gen Intel i5-12450H, Intel integrated graphics, no NVIDIA GPU.
- Keep the final model artifact at or below 300 MB. Runtime memory is a separate measurement.
- No paid endpoint, cloud GPU, or provider inference is permitted without explicit approval.

## Local setup

Use a virtual environment so training dependencies stay separate from the application:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

## Data gate

Do not start training until a corpus has documented source, license, permitted use, and split plan. See `data/README.md`. The starter trainer expects `data/raw/corpus.txt`; that file is intentionally absent.

After data approval and installation, a short pipeline smoke test can be run with:

```powershell
python -m scratch_lm.train --data data/raw/corpus.txt --steps 10 --batch-size 2
```

The model currently learns next-byte prediction only. It does not yet score interview answers. An interview scoring dataset, rubric, instruction-training stage, held-out human evaluation, and a viable no-cost remote host remain required gates from the SRS.
