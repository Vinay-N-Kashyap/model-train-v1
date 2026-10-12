# Data policy and corpus intake

No corpus is included yet. Do not place student transcripts, private data, or downloaded datasets here until the project owner has checked licensing, consent, retention, and the data split plan.

The trainer refuses to run unless `data/raw/corpus.manifest.json` documents the corpus and passes the gate. Start from `corpus.manifest.example.json`, but do not copy its placeholder values unchanged.

For any proposed corpus, record:

- Dataset name, source URL, exact version/date, license, and permitted use.
- Whether it contains personal or sensitive data and how it was removed.
- A short description of domains, language, and known limitations.
- The exact train/validation/test split plan and duplicate checks.
- An explicit project-owner confirmation that the data license permits model training.
- Whether any retained personal data has a documented consent/legal basis and retention policy.

The trainer expects a UTF-8 plain-text file at `data/raw/corpus.txt` and a matching provenance manifest. It uses the manifest's sequential byte split into train, validation, and held-out test segments; this is a baseline split, not a substitute for document-level de-duplication or a leakage review. The starter byte-level model is a pipeline feasibility baseline only; it is not expected to provide useful interview evaluation quality.

