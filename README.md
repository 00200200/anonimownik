Animownik – quick guide

Requirements

- Python 3.11+
- UV
- Install deps from `pyproject.toml`: `uv sync .`
- spaCy model: `python -m spacy download pl_core_news_sm`
- NER model (default `models/herbert-ner`) – provide your own or train with `src/scripts/train.py`

How to run

- Single text:  
  `python -m src.scripts.run_pipeline --text "Example text" --model models/herbert-ner`
- Input file (writes JSON and TXT next to output path):  
  `python -m src.scripts.run_pipeline --file data/processed/test.txt --output outputs/result --model models/herbert-ner`
- Interactive mode (keyboard input):  
  `python -m src.scripts.run_pipeline --model models/herbert-ner`

What the pipeline does

- Module 1: anonymizes text, stores entity metadata and morphological features (spaCy).
- Module 2: generates synthetic substitutions for detected token.
