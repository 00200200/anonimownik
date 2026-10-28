# Anonimownik - pipeline anonimizacji dla języka polskiego

**Zespół: SolvroGen**

Pipeline NER (HerBERT) + RegEx do zastępowania wykrytych danych w tekstach polskich tagami lub danymi syntetycznymi.

## O projekcie

Anonimownik to prototyp zespołu SolvroGen z hackathonu HackNation 2025. Łączy model NER oparty na HerBERT z regułami RegEx. Generator podstawień korzysta z cech morfologicznych spaCy, słowników odmienionych form i heurystyk.

Repozytorium zawiera skrypt pomiaru precision, recall i F1 dla NER, ale nie publikuje jego wyników ani oceny poprawności gramatycznej generatora. Raport `docs/performance_solvrogen.txt` opisuje czas przetwarzania, nie jakość wykrywania. Pipeline może pomijać dane i tworzyć błędne odmiany; wynik wymaga sprawdzenia przed udostępnieniem i nie gwarantuje anonimowości.

## Szybki start

1. **Pobierz model NER** z [Google Drive](https://drive.google.com/file/d/1EqvHlcNaYqUdA3drFEuIjB-uFH_F87ho/view?usp=drive_link)
2. **Rozpakuj model** do katalogu `models/herbert-ner/`
3. **Zainstaluj zależności**: `uv sync` (lub `pip install -e .`)
4. **Zainstaluj model spaCy**: `uv run python -m spacy download pl_core_news_sm` (przy instalacji przez pip: `python -m spacy download pl_core_news_sm`)
5. **Uruchom**: `uv run python -m src.pipeline`

Szczegółowe instrukcje znajdziesz w sekcji [Instalacja](#instalacja) i [Uruchomienie](#uruchomienie).

## Struktura repozytorium

```
anonimownik/
├── src/
│   ├── pipeline/              # Główny pipeline anonimizacji
│   │   ├── anonymizer.py      # Moduł anonimizacji (NER + regex)
│   │   ├── synthetic_generator.py  # Generator syntetycznych danych
│   │   ├── pipeline.py        # Połączony pipeline
│   │   ├── cli.py             # Interfejs CLI
│   │   └── __main__.py        # Punkt wejścia python -m src.pipeline
│   └── scripts/
│       ├── train.py           # Trenowanie modelu NER
│       ├── evaluate.py        # Ewaluacja modelu
│       └── run_pipeline.py    # Uruchamianie pipeline'u
├── data/                      # Dane treningowe
├── models/                    # Wytrenowane modele (wagi HerBERT-NER)
├── generate_output.py         # Zapis tekstu z tagami, linia po linii
├── docs/
│   ├── output_SolvroGen.txt    # Plik wynikowy z konkursu
│   ├── performance_solvrogen.txt  # Czas przetwarzania i opis sprzętu
│   ├── preprocessing_solvrogen.md # Opis przetwarzania danych
│   └── synthetic_generation_solvrogen.md # Opis generacji syntetycznej
└── pyproject.toml             # Konfiguracja projektu
```

## Jak to działa

### Proces anonimizacji

1. **Tokenizacja** - NER używa tokenizera HerBERT; spaCy służy do analizy morfologicznej
2. **Analiza morfologiczna** - dla wykrytych encji zbierane są dostępne cechy (przypadek, rodzaj, liczba)
3. **Wykrywanie encji** - hybrydowe podejście:
   - **Model NER (HerBERT)** - wykrywa imiona, nazwiska, miasta, firmy itp.
   - **Reguły RegEx** - wykrywa PESEL, email, telefon, nr konta, nr dokumentu
4. **Zamiana na tagi** - encje zastępowane są tagami `[nazwa_typu]`

### Założenia

- Tagi mają format: `[name]`, `[surname]`, `[city]`, `[phone]`, `[pesel]`, `[email]` itp.
- `generate_output.py` przetwarza tekst linia po linii; główny CLI `--input` dzieli plik na akapity i zapisuje raport
- Dostępne cechy morfologiczne służą do wyboru podstawień; brakujące formy mogą zostać zastąpione mianownikiem
- NER przetwarza maksymalnie 512 tokenów jednego fragmentu; dalsza część jest pomijana przez model, choć RegEx nadal analizuje cały tekst
- Bez istniejącego katalogu modelu NER pipeline może działać tylko z RegEx, bez wykrywania encji przez HerBERT

### Co robi pipeline

- **Moduł 1**: anonimizuje tekst, przechowuje metadane encji i cechy morfologiczne (spaCy)
- **Moduł 2**: generuje syntetyczne podstawienia dla wykrytych tokenów

## Instalacja

### Wymagania

- Python 3.11+
- UV (opcjonalnie, można użyć pip)
- PyTorch >= 2.0.0
- spaCy >= 3.7.0
- transformers >= 4.37.0

### Instalacja zależności

```bash
# Klonowanie repozytorium
git clone https://github.com/00200200/anonimownik.git
cd anonimownik

# Instalacja zależności z uv (zalecane)
uv sync

# Lub z pip:
pip install -e .

# Instalacja modelu spaCy dla polskiego
uv run python -m spacy download pl_core_news_sm
# Przy instalacji przez pip: python -m spacy download pl_core_news_sm

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

### Instalacja z Dockerem

Konfiguracja wymaga uwagi: `Dockerfile` instaluje projekt przed skopiowaniem `README.md` i źródeł, a instrukcja kopiowania `src/` jest zakomentowana. Budowanie może wymagać poprawienia Dockerfile; przykłady uruchomienia poniżej zakładają zbudowany obraz i montują źródła oraz model. Lokalny CLI opisano wyżej.

```bash
# 1. Pobierz model NER z Google Drive (zobacz sekcję Instalacja powyżej)
#    i umieść go w katalogu models/herbert-ner/

# 2. Budowanie obrazu
docker build -t animownik:latest .

# Lub użyj docker-compose
docker-compose build

# Uwaga: Model NER musi być dostępny w kontenerze.
# Docker-compose automatycznie montuje katalog ./models do /app/models
```

## Uruchomienie

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
# Raport z tekstem oryginalnym, tekstem z tagami i podstawieniami syntetycznymi
uv run python -m src.pipeline --input data/input.txt --output outputs/result

# Z określeniem formatu wyjściowego (json, text, both)
uv run python -m src.pipeline --input data/input.txt --output outputs/result --format both

# Tylko anonimizacja (bez generacji syntetycznej)
uv run python generate_output.py input.txt output_solvrogen.txt

# Przetwarzanie z pomiarem czasu
time uv run python generate_output.py input.txt output_solvrogen.txt
```

Eksporty `text`, `json` i `both` głównego CLI zawierają **oryginalny tekst**; JSON zawiera również oryginalne wartości encji. Flaga `--anonymize-only` nie zmienia eksportu plików. Aby zapisać sam tekst z tagami, użyj `generate_output.py` jak wyżej albo zapisz wyłącznie `result.anon_text` z API. Także taki tekst może zawierać niewykryte dane.

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
# Przetwarzanie tekstu (entrypoint src.scripts.run_pipeline przyjmuje tekst pozycyjnie)
docker run --rm \
  -v "$(pwd)/src:/app/src:ro" \
  -v "$(pwd)/models:/app/models:ro" \
  animownik:latest "Jan Kowalski mieszka w Warszawie"

# Przetwarzanie pliku
docker run --rm \
  -v "$(pwd)/src:/app/src:ro" \
  -v "$(pwd)/models:/app/models:ro" \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/outputs:/app/outputs \
  animownik:latest \
  --file /app/data/input.txt \
  --output /app/outputs/result.txt

# Użycie docker-compose
docker-compose run --rm -v "$(pwd)/src:/app/src:ro" animownik "Test"
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

## Przykładowe typy encji

| Typ | Opis | Metoda wykrywania |
|-----|------|-------------------|
| `[name]` | Imię | NER |
| `[surname]` | Nazwisko | NER |
| `[city]` | Miasto | NER |
| `[address]` | Adres | NER |
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

## Pliki konkursu

| Plik | Opis |
|------|------|
| `README.md` | Ten plik - dokumentacja projektu |
| `docs/output_SolvroGen.txt` | Plik wynikowy z konkursu |
| `docs/performance_solvrogen.txt` | Czas przetwarzania i opis sprzętu |
| `docs/preprocessing_solvrogen.md` | Opis procesu zbierania i przetwarzania danych |
| `docs/synthetic_generation_solvrogen.md` | Opis metody generacji danych syntetycznych |

## Architektura

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

## Zespół SolvroGen

Projekt stworzony na hackathon HackNation 2025.
