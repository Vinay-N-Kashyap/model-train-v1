# Data policy and corpus intake

No corpus is included yet. Do not place student transcripts, private data, or downloaded datasets here until the project owner has checked licensing, consent, retention, and the data split plan.

For any proposed corpus, record:

- Dataset name, source URL, exact version/date, license, and permitted use.
- Whether it contains personal or sensitive data and how it was removed.
- A short description of domains, language, and known limitations.
- The exact train/validation/test split manifest and duplicate checks.

The current trainer expects a UTF-8 plain-text file at `data/raw/corpus.txt`. The starter byte-level model is a pipeline feasibility baseline only; it is not expected to provide useful interview evaluation quality.

