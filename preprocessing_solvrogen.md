# Preprocessing - Zespół SolvroGen

## Opis procesu przetwarzania danych

### 1. Źródła danych treningowych

Dane do trenowania modelu NER pochodzą z:

1. **Dane NASK** - oficjalny zbiór treningowy dostarczony przez organizatorów
   - Lokalizacja: `data/nask_train/`
   - Format: pliki tekstowe z oznaczonymi encjami

2. **Dane syntetyczne** - wygenerowane przez nasz generator
   - Zwiększenie różnorodności przykładów
   - Pokrycie edge case'ów

### 2. Czyszczenie danych

#### 2.1 Normalizacja tekstu

```python
# Usuwanie wielokrotnych spacji
text = re.sub(r'\s+', ' ', text)

# Normalizacja znaków specjalnych
text = text.replace('–', '-').replace('—', '-')

# Usuwanie znaków kontrolnych
text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
```

#### 2.2 Obsługa formatów danych

- **Wieloliniowe dokumenty** - dzielone na pojedyncze zdania/akapity
- **Dokumenty OCR** - zachowanie oryginalnego formatowania (literówki, błędy)
- **Dane tabelaryczne** - specjalna obsługa struktur typu "Imię: Jan"

### 3. Augmentacja danych

Dla zwiększenia robustności modelu stosujemy:

1. **Zamiana synonimów kontekstowych**
   - "Mieszkam w Warszawie" → "Mieszkam w stolicy"
   - Zachowanie poprawności gramatycznej

2. **Perturbacje tekstowe**
   - Losowe błędy literowe (symulacja OCR)
   - Zmiana wielkości liter
   - Dodawanie/usuwanie interpunkcji

3. **Generacja nowych przykładów**
   - Użycie szablonów zdań
   - Losowe wypełnianie encjami ze słowników

### 4. Konwersja do formatu treningowego

#### Format BIO (Beginning-Inside-Outside)

```
Jan       B-name
Kowalski  B-surname
mieszka   O
w         O
Warszawie B-city
.         O
```

#### Mapowanie etykiet

| Etykieta | Znaczenie |
|----------|-----------|
| B-name | Początek imienia |
| I-name | Kontynuacja imienia |
| B-surname | Początek nazwiska |
| B-city | Miasto |
| B-address | Adres |
| B-email | Email |
| B-phone | Telefon |
| B-pesel | PESEL |
| B-document-number | Nr dokumentu |
| O | Inne (nie-encja) |

### 5. Podział danych

```
Dane treningowe: 80%
Dane walidacyjne: 10%
Dane testowe: 10%
```

### 6. Statystyki zbioru

| Typ encji | Liczba wystąpień |
|-----------|------------------|
| name | ~5000 |
| surname | ~4500 |
| city | ~3000 |
| address | ~2000 |
| phone | ~2500 |
| email | ~1500 |
| pesel | ~1000 |
| document-number | ~800 |
| company | ~600 |
| date | ~2000 |

### 7. Problemy i rozwiązania

#### Problem: Nakładające się encje
```
"Jan Kowalski z Warszawy" - czy "Jan Kowalski" to jedna encja czy dwie?
```
**Rozwiązanie**: Traktujemy imię i nazwisko jako osobne encje (B-name, B-surname)

#### Problem: Adresy z wieloma komponentami
```
"ul. Długa 5/12, 00-001 Warszawa"
```
**Rozwiązanie**: Cały adres jako jedna encja [address], miasto osobno jeśli wyraźnie oddzielone

#### Problem: Numery telefonów w różnych formatach
```
+48 500 600 700
500-600-700
(22) 123-45-67
```
**Rozwiązanie**: Reguły RegEx pokrywające wszystkie popularne formaty polskich numerów

### 8. Pipeline przetwarzania

```
Surowy tekst
    │
    ▼
Normalizacja znaków
    │
    ▼
Tokenizacja (spaCy)
    │
    ▼
Analiza morfologiczna
    │
    ▼
Konwersja do formatu BIO
    │
    ▼
Zapis do pliku treningowego
```

### 9. Narzędzia użyte w preprocessingu

- **Python 3.11+** - język główny
- **spaCy** - tokenizacja i analiza morfologiczna
- **regex** - zaawansowane wyrażenia regularne
- **pandas** - manipulacja danymi tabelarycznymi

