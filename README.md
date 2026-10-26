# Spam filtering: classic model vs Transformer, from a PC to a Raspberry Pi 5

[![tests](https://github.com/amine-belkacem/spam-classic-vs-transformer-edge/actions/workflows/tests.yml/badge.svg)](https://github.com/amine-belkacem/spam-classic-vs-transformer-edge/actions/workflows/tests.yml)

**Work in progress.** No results yet: this page only describes the plan and how to get the data.
Numbers will appear here once they are measured, with the method used to get them.

## The question

A fine-tuned Transformer usually wins on accuracy. But a spam filter on an SMS
gateway or a small ARM board has to answer fast, in little memory, without a GPU.
I want to know, with measurements, when TF-IDF + logistic regression is enough
and when DistilBERT is worth its cost, and whether a cascade (cheap model first,
DistilBERT only for unsure messages) gets the best of both.

The full plan, requirements and success criteria are in [docs/specification.md](docs/specification.md).

## Getting the data

```bash
python -m venv .venv
.venv/Scripts/activate      # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
pytest -v
```

The script downloads both datasets from UCI, checks the SHA-256 of every file
and compares the row and spam counts with the ones UCI publishes.

## Data and credits

- **SMS Spam Collection**: Almeida, T. & Hidalgo, J. (2011). UCI Machine Learning Repository.
  https://doi.org/10.24432/C5CC84. CC BY 4.0.
- **YouTube Spam Collection**: Alberto, T. & Lochter, J. (2015). UCI Machine Learning Repository.
  https://doi.org/10.24432/C58885. CC BY 4.0.

Neither dataset is redistributed here. The YouTube files include comment authors'
names; the loader does not read that column.

## License

Code under the MIT license (see [LICENSE](LICENSE)).
