# Animownik - Pipeline Anonimizacji dla Języka Polskiego

**Zespół: SolvroGen**

> Hybrydowe rozwiązanie NER (HerBERT) + RegEx do anonimizacji tekstów polskich z obsługą odmiany morfologicznej.

## 📋 O projekcie

Animownik to pipeline do automatycznej anonimizacji danych wrażliwych (PII) w tekstach polskich. Łączy model NER oparty na HerBERT z regułami RegEx, zapewniając wysoką skuteczność wykrywania encji przy zachowaniu poprawności gramatycznej w danych syntetycznych.

## ⚡ Szybki start

1. **Pobierz model NER** z [Google Drive](https://drive.google.com/file/d/1EqvHlcNaYqUdA3drFEuIjB-uFH_F87ho/view?usp=drive_link)
2. **Rozpakuj model** do katalogu `models/herbert-ner/`
3. **Zainstaluj zależności**: `uv sync` (lub `pip install -e .`)
4. **Zainstaluj spaCy**: `python -m spacy download pl_core_news_sm`
5. **Uruchom**: `uv run python -m src.pipeline`

Szczegółowe instrukcje znajdziesz w sekcji [Instalacja](#-instalacja) i [Uruchomienie](#-uruchomienie).

## 🗂️ Struktura repozytorium

```
animownik/
├── src/
│   ├── pipeline/              # Główny pipeline anonimizacji
│   │   ├── anonymizer.py      # Moduł anonimizacji (NER + regex)
│   │   ├── synthetic_generator.py  # Generator syntetycznych danych
│   │   ├── pipeline.py        # Połączony pipeline
│   │   ├── cli.py             # Interfejs CLI
│   │   └── README.md          # Szczegółowa dokumentacja
│   └── scripts/
│       ├── train.py           # Trenowanie modelu NER
│       ├── evaluate.py        # Ewaluacja modelu
│       └── run_pipeline.py    # Uruchamianie pipeline'u
├── data/                      # Dane treningowe
├── models/                    # Wytrenowane modele (wagi HerBERT-NER)
├── output_solvrogen.txt       # Zanonimizowany plik wynikowy
├── performance_solvrogen.txt  # Metryki wydajności
├── preprocessing_solvrogen.md # Opis przetwarzania danych
├── synthetic_generation_solvrogen.md  # Opis generacji syntetycznej
└── pyproject.toml             # Konfiguracja projektu
```

## 🔧 Jak to działa

### Proces anonimizacji

1. **Tokenizacja** - tekst jest dzielony na tokeny przy użyciu spaCy
2. **Analiza morfologiczna** - dla każdego tokenu zapisywane są cechy morfologiczne (przypadek, rodzaj, liczba)
3. **Wykrywanie encji** - hybrydowe podejście:
   - **Model NER (HerBERT)** - wykrywa imiona, nazwiska, miasta, firmy itp.
   - **Reguły RegEx** - wykrywa PESEL, email, telefon, nr konta, nr dokumentu
4. **Zamiana na tagi** - encje zastępowane są tagami `[nazwa_typu]`

### Założenia

- Tagi mają format: `[name]`, `[surname]`, `[city]`, `[phone]`, `[pesel]`, `[email]` itp.
- Każda linia wejściowa odpowiada jednej linii wyjściowej (1:1)
- Cechy morfologiczne są zachowywane dla generacji syntetycznej

### Co robi pipeline

- **Moduł 1**: anonimizuje tekst, przechowuje metadane encji i cechy morfologiczne (spaCy)
- **Moduł 2**: generuje syntetyczne podstawienia dla wykrytych tokenów

## 🚀 Instalacja

### Wymagania

- Python 3.11+
- UV (opcjonalnie, można użyć pip)
- PyTorch >= 2.0.0
- spaCy >= 3.7.0
- transformers >= 4.37.0

### Instalacja zależności

```bash
# Klonowanie repozytorium
git clone https://github.com/your-repo/animownik.git
cd animownik

# Instalacja zależności z uv (zalecane)
uv sync .

# Lub z pip:
pip install -e .

# Instalacja modelu spaCy dla polskiego
python -m spacy download pl_core_news_sm

# Pobranie modelu NER
# Model HerBERT-NER jest wymagany do działania pipeline'u.
# Pobierz model z Google Drive:
# https://drive.google.com/file/d/1EqvHlcNaYqUdA3drFEuIjB-uFH_F87ho/view?usp=drive_link
#
# Po pobraniu:
# 1. Rozpakuj archiwum
# 2. Umieść zawartość w katalogu models/herbert-ner/
#    (powinien zawierać pliki: config.json, pytorch_model.bin, tokenizer_config.json, itp.)
#
# Przykład struktury:
# models/
# └── herbert-ner/
#     ├── config.json
#     ├── pytorch_model.bin
#     ├── tokenizer_config.json
#     ├── vocab.txt
#     └── labels.json
#
# Alternatywnie: wytrenuj własny model używając:
# python src/scripts/train.py
```

### Instalacja z Dockerem (zalecane)

```bash
# 1. Pobierz model NER z Google Drive (zobacz sekcję Instalacja powyżej)
#    i umieść go w katalogu models/herbert-ner/

# 2. Budowanie obrazu
docker build -t animownik:latest .

# Lub użyj docker-compose
docker-compose build

# Uwaga: Model NER musi być dostępny w kontenerze.
# Docker-compose automatycznie montuje katalog ./models do /app/models
# Szczegółowa dokumentacja: DOCKER.md
```

## 💻 Uruchomienie

### Szybki start

```bash
# 1. Upewnij się, że model NER jest pobrany i umieszczony w models/herbert-ner/
#    (zobacz sekcję Instalacja powyżej)

# 2. Uruchom pipeline w trybie interaktywnym
uv run python -m src.pipeline

# Lub bezpośrednio:
python -m src.pipeline
```

### Przetwarzanie pojedynczego tekstu

```bash
# Z użyciem uv (zalecane)
uv run python -m src.pipeline --text "Jan Kowalski mieszka w Warszawie"

# Lub bezpośrednio
python -m src.pipeline --text "Jan Kowalski mieszka w Warszawie" --model models/herbert-ner
```

### Przetwarzanie pliku

```bash
# Anonimizacja pliku wejściowego
uv run python -m src.pipeline --input data/input.txt --output outputs/result

# Z określeniem formatu wyjściowego (json, text, both)
uv run python -m src.pipeline --input data/input.txt --output outputs/result --format both

# Tylko anonimizacja (bez generacji syntetycznej)
uv run python -m src.pipeline --input input.txt --output output_solvrogen.txt --anonymize-only

# Przetwarzanie z pomiarem czasu
time uv run python -m src.pipeline --input input.txt --output output_solvrogen.txt --anonymize-only
```

### Tryb interaktywny (wejście z klawiatury)

```bash
# Najprostszy sposób - uruchom bez argumentów
uv run python -m src.pipeline

# Lub z podaniem modelu
python -m src.pipeline --model models/herbert-ner
```

Po uruchomieniu wpisz tekst do przetworzenia lub `q` aby wyjść.

### Uruchomienie z Dockerem

```bash
# Przetwarzanie tekstu
docker run --rm animownik:latest --text "Jan Kowalski mieszka w Warszawie"

# Przetwarzanie pliku
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/outputs:/app/outputs \
  animownik:latest \
  --file /app/data/input.txt \
  --output /app/outputs/result.txt

# Użycie skryptu pomocniczego
./docker-run.sh --text "Test"

# Użycie docker-compose
docker-compose run --rm animownik --text "Test"

# Szczegółowa dokumentacja: DOCKER.md
```

### Użycie programatyczne

```python
from src.pipeline import create_pipeline

# Utwórz pipeline
pipeline = create_pipeline(model_path="models/herbert-ner")

# Anonimizuj tekst
result = pipeline.anonymize_only("Jan Kowalski mieszka w Warszawie. PESEL: 90010112345")
print(result.anon_text)
# "[name] [surname] mieszka w [city]. PESEL: [pesel]"
```

## 📊 Obsługiwane typy encji

| Typ | Opis | Metoda wykrywania |
|-----|------|-------------------|
| `[name]` | Imię | NER |
| `[surname]` | Nazwisko | NER |
| `[city]` | Miasto | NER |
| `[address]` | Adres | NER + RegEx |
| `[email]` | Email | RegEx |
| `[phone]` | Telefon | RegEx |
| `[pesel]` | PESEL | RegEx (11 cyfr) |
| `[document-number]` | Nr dokumentu | RegEx |
| `[bank-account]` | Nr konta bankowego | RegEx (26 cyfr) |
| `[credit-card-number]` | Nr karty kredytowej | RegEx |
| `[company]` | Firma | NER |
| `[job-title]` | Stanowisko | NER |
| `[date]` | Data | RegEx |
| `[age]` | Wiek | NER |
| `[sex]` | Płeć | NER |

## 📁 Pliki konkursu

| Plik | Opis |
|------|------|
| `README.md` | Ten plik - dokumentacja projektu |
| `output_solvrogen.txt` | Zanonimizowany plik wynikowy |
| `performance_solvrogen.txt` | Czas przetwarzania i opis sprzętu |
| `preprocessing_solvrogen.md` | Opis procesu zbierania i przetwarzania danych |
| `synthetic_generation_solvrogen.md` | Opis metody generacji danych syntetycznych |
| `presentation_solvrogen.pdf` | Prezentacja projektu (max 5 slajdów) |

## 🏗️ Architektura

```
┌─────────────────────────────────────────────────────────────┐
│                    AnonymizationPipeline                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │     Anonymizer      │    │   SyntheticGenerator        │ │
│  │  ┌───────────────┐  │    │  ┌───────────────────────┐  │ │
│  │  │  HerBERT NER  │  │───►│  │  Słowniki + odmiana   │  │ │
│  │  └───────────────┘  │    │  │  morfologiczna        │  │ │
│  │  ┌───────────────┐  │    │  └───────────────────────┘  │ │
│  │  │  RegEx Rules  │  │    │  ┌───────────────────────┐  │ │
│  │  └───────────────┘  │    │  │  spaCy (morfologia)   │  │ │
│  │  ┌───────────────┐  │    │  └───────────────────────┘  │ │
│  │  │  spaCy Morph  │  │    └─────────────────────────────┘ │
│  │  └───────────────┘  │                                    │
│  └─────────────────────┘                                    │
└─────────────────────────────────────────────────────────────┘
```

## 👥 Zespół SolvroGen

Projekt stworzony na hackathon HackNation 2025.

## 📄 Licencja

MIT License
