# TauraBot

> *Taura* means "speak" in Shona.

TauraBot is a research / portfolio project building the **first openly released fine-tuned language model for Shona**, a Bantu language spoken by ~15 million people that has almost no production NLP tooling today.

The project releases four artifacts on HuggingFace:

1. **Dataset** — the largest openly available cleaned Shona text corpus
2. **Model** — `shona-mt5-small`, mT5-small continued-pretrained on the Shona corpus
3. **Model** — `taurabot-shona`, the conversation fine-tuned chatbot
4. **Space** — a Gradio demo at `[username]/taurabot`

## Status

**Phase 1: Data Collection & Corpus Building — in progress**

| Phase | Description | Status |
|------|-------------|--------|
| 1 | Build cleaned Shona corpus (≥1M sentences) | 🚧 in progress |
| 2 | Continued pretraining of mT5-small | ⏳ planned |
| 3 | Conversation fine-tuning on hand-crafted Shona pairs | ⏳ planned |
| 4 | Gradio chatbot on HuggingFace Spaces | ⏳ planned |

## Project structure

```
taurabot/
├── data/
│   ├── raw/                # Raw downloads (gitignored)
│   ├── processed/          # Cleaned corpus files
│   └── conversations/      # Shona conversation pairs (JSON)
├── notebooks/              # Walkthrough notebooks per phase
├── src/
│   ├── data/               # Scrapers + cleaning pipeline
│   ├── model/              # Pretraining + fine-tuning
│   └── chatbot/            # Inference logic
├── configs/config.yaml     # All hyperparameters (no hardcoding)
├── app.py                  # Gradio frontend
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Corpus statistics

*To be filled in after Phase 1 completes.*

## Model performance

*To be filled in after Phase 2 / 3 complete.*

## Limitations

*To be filled in after evaluation.*

## Citation

*BibTeX block to be added when the dataset + model are released.*
