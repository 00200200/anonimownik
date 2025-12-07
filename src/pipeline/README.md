# Pipeline Anonimizacji i Generacji Syntetycznej

Pipeline do przetwarzania tekstów polskich składający się z dwóch modułów:

## Moduł 1 - Anonimizacja (`Anonymizer`)

Wykrywa encje wrażliwe w tekście i zapisuje ich metadane morfologiczne.

### Funkcjonalność

1. **Tokenizacja i tagowanie morfosyntaktyczne** (spaCy)
2. **NER + reguły** - wykrywanie encji za pomocą modelu NER i reguł regexowych
3. **Zapis metadanych**:
   - Span w tekście (offsety)
   - Typ encji
   - Oryginalny tekst
   - Cechy morfologiczne (przypadek, liczba, rodzaj)

### Użycie

```python
from src.pipeline import Anonymizer

# Bez modelu NER (tylko reguły regex)
anonymizer = Anonymizer(model_path=None)

# Z modelem NER
anonymizer = Anonymizer(model_path="models/herbert-ner")

# Anonimizacja
text = "Anna mieszka w Warszawie. Jej PESEL to 90010112345."
result = anonymizer.anonymize(text)

print(result.anon_text)
# "{name} mieszka w {city}. Jej PESEL to {pesel}."

print(result.entities_meta)
# [EntityMeta(start=0, end=4, entity_type='name', original_text='Anna', 
#             morph_features={'case': 'Nom', 'gender': 'Fem', 'number': 'Sing'}), ...]
```

### Obsługiwane typy encji

- `name`, `surname` - imiona i nazwiska
- `city`, `address` - lokalizacje
- `email`, `phone` - dane kontaktowe
- `pesel`, `document-number` - dokumenty
- `bank-account`, `credit-card-number` - dane finansowe
- `political-view`, `ethnicity`, `religion` - dane wrażliwe
- `company`, `school-name`, `job-title` - praca/edukacja
- i więcej...

---

## Moduł 2 - Generator Syntetyczny (`SyntheticGenerator`)

Generuje nowe wartości i odmienia je zgodnie z cechami morfologicznymi.

### Funkcjonalność

1. **Losowanie nowych wartości** z odpowiednich słowników
2. **Odmiana morfologiczna** zgodnie z zapisanymi cechami
3. **Podmiana tokenów** w zanonimizowanym tekście

### Użycie

```python
from src.pipeline import SyntheticGenerator

generator = SyntheticGenerator(seed=42)

# Prosty tryb (mianownik)
anon_text = "Nazywam się {name} {surname}."
result = generator.generate_from_anon_text(anon_text)
# "Nazywam się Anna Kowalska."

# Z metadanymi (poprawna odmiana)
anon_text = "To jest dom {name}."
entities_meta = [{
    "entity_type": "name",
    "original_text": "Anny",
    "morph_features": {"case": "Gen", "gender": "Fem"}
}]
result = generator.generate(anon_text, entities_meta)
# "To jest dom Aleksandry."
```

### Obsługiwane przypadki

| Przypadek | Skrót | Przykład |
|-----------|-------|----------|
| Mianownik | nom | Anna, Jan |
| Dopełniacz | gen | Anny, Jana |
| Celownik | dat | Annie, Janowi |
| Biernik | acc | Annę, Jana |
| Narzędnik | inst | Anną, Janem |
| Miejscownik | loc | Annie, Janie |

---

## Pełny pipeline

### Programatycznie

```python
from pipeline import AnonymizationPipeline, create_pipeline

# Utwórz pipeline
pipeline = create_pipeline(seed=42)  # automatycznie szuka modelu

# Przetwórz tekst
text = "Anna Kowalska mieszka w Warszawie."
result = pipeline.process(text)

print(result.original_text)     # "Anna Kowalska mieszka w Warszawie."
print(result.anonymized_text)   # "{name} {surname} mieszka w {city}."
print(result.synthetic_text)    # "Magdalena Wiśniewska mieszka w Krakowie."
print(result.entities)          # [EntityMeta(...), ...]
print(result.entity_counts)     # {"name": 1, "surname": 1, "city": 1}
```

### Przetwarzanie plików

```python
from pipeline import create_pipeline

pipeline = create_pipeline()

# Pojedynczy plik
result = pipeline.process_file(
    "dane.txt",
    "wynik.json",
    output_format="json"  # lub "text", "both"
)

# Katalog
results = pipeline.process_directory(
    "dokumenty/",
    "wyniki/",
    pattern="*.txt"
)
```

### Batch processing

```python
texts = [
    "Jan Kowalski mieszka w Warszawie.",
    "Anna Nowak pracuje w CD Projekt.",
    "Kontakt: jan@example.com, tel. 500-123-456."
]

result = pipeline.process_batch(texts)
print(f"Przetworzono {len(result.results)} tekstów")
print(f"Wykryto {result.total_entities} encji")
```

---

## CLI (Command Line Interface)

Pipeline można uruchomić z linii poleceń:

```bash
# Tryb interaktywny
python -m pipeline

# Przetworzenie tekstu
python -m pipeline --text "Jan Kowalski mieszka w Warszawie."

# Przetworzenie pliku
python -m pipeline --input dane.txt --output wynik.json --format json

# Przetworzenie katalogu
python -m pipeline --input-dir dokumenty/ --output-dir wyniki/

# Tylko anonimizacja (bez generacji syntetycznej)
python -m pipeline --text "Jan Kowalski" --anonymize-only

# Z własnym modelem NER
python -m pipeline --model models/herbert-ner --text "Test"

# Więcej informacji
python -m pipeline --verbose --text "Jan Kowalski z Warszawy"
```

### Opcje CLI

| Opcja | Opis |
|-------|------|
| `--text, -t` | Tekst do przetworzenia |
| `--input, -i` | Plik wejściowy |
| `--input-dir` | Katalog wejściowy |
| `--output, -o` | Plik wyjściowy |
| `--output-dir` | Katalog wyjściowy |
| `--format, -f` | Format wyjścia (json/text/both) |
| `--model, -m` | Ścieżka do modelu NER |
| `--seed, -s` | Ziarno losowości |
| `--no-regex` | Wyłącz reguły regex |
| `--anonymize-only` | Tylko anonimizacja |
| `--verbose, -v` | Szczegółowe informacje |
| `--json` | Wynik jako JSON |

---

## Wymagania

```bash
pip install spacy torch transformers
python -m spacy download pl_core_news_sm
```

## Użycie CLI

```bash
# Uruchomienie pipeline'u
python -m src.scripts.run_pipeline "Jan Kowalski mieszka w Warszawie."

# Tryb interaktywny
python -m src.scripts.run_pipeline
```
