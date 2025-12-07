# Generacja Danych Syntetycznych - Zespół SolvroGen

## Opis metody

### Mechanizm generacji

Nasz generator syntetyczny (`SyntheticGenerator`) zamienia tagi typu `{name}`, `{city}` na sensowne, odmienione wartości, korzystając z:

1. **Słowników z odmianami** - predefiniowane listy imion, nazwisk, miast z wszystkimi przypadkami
2. **Analiza morfologiczna spaCy** - wykrywanie przypadka z kontekstu
3. **Metadane z anonimizacji** - cechy morfologiczne oryginalnej encji

### Źródła danych syntetycznych

| Typ encji | Źródło |
|-----------|--------|
| Imiona | Słownik ~30 popularnych polskich imion (męskie + żeńskie) |
| Nazwiska | Słownik ~20 popularnych polskich nazwisk |
| Miasta | 10 największych polskich miast |
| Adresy | Szablony ulic + losowe numery |
| Email | Generowane z imię.nazwisko@domena |
| Telefon | Losowe numery w formacie +48 XXX XXX XXX |
| PESEL | Losowe 11 cyfr |
| Nr konta | Losowe 26 cyfr w formacie bankowym |

---

## Walka z fleksją - kluczowy element

### Problem odmiany w języku polskim

Język polski ma 7 przypadków, co oznacza że każde imię, nazwisko czy miasto musi być poprawnie odmienione:

| Przypadek | Przykład (miasto) | Użycie |
|-----------|-------------------|--------|
| Mianownik | Warszawa | Warszawa jest piękna |
| Dopełniacz | Warszawy | Jestem z Warszawy |
| Celownik | Warszawie | Jadę do Warszawy |
| Biernik | Warszawę | Odwiedzam Warszawę |
| Narzędnik | Warszawą | Zachwycam się Warszawą |
| Miejscownik | Warszawie | Mieszkam w Warszawie |
| Wołacz | Warszawo! | O Warszawo! |

### Nasze rozwiązanie

#### Krok 1: Zapisanie cech morfologicznych podczas anonimizacji

```python
# W module Anonymizer
morph_features = {
    'case': 'Loc',      # Miejscownik
    'gender': 'Fem',    # Rodzaj żeński
    'number': 'Sing'    # Liczba pojedyncza
}
```

#### Krok 2: Słowniki z pełną odmianą

```python
CITIES = {
    "Warszawa": {
        "nom": "Warszawa",
        "gen": "Warszawy", 
        "dat": "Warszawie",
        "acc": "Warszawę",
        "inst": "Warszawą",
        "loc": "Warszawie"
    },
    "Kraków": {
        "nom": "Kraków",
        "gen": "Krakowa",
        "dat": "Krakowowi",
        "acc": "Kraków",
        "inst": "Krakowem",
        "loc": "Krakowie"
    },
    # ... więcej miast
}
```

#### Krok 3: Dobór formy na podstawie kontekstu

```python
def _get_inflected_city(self, case: str) -> str:
    city_dict = random.choice(list(self.cities.values()))
    return city_dict.get(case, city_dict.get('nom'))
```

### Wykrywanie przypadka

Jeśli metadane morfologiczne nie są dostępne, analizujemy kontekst:

```python
# Analiza przyimków
PREPOSITION_CASES = {
    'w': 'loc',      # w Warszawie (miejscownik)
    'do': 'gen',     # do Warszawy (dopełniacz)  
    'z': 'gen',      # z Warszawy (dopełniacz)
    'dla': 'gen',    # dla Warszawy (dopełniacz)
    'przez': 'acc',  # przez Warszawę (biernik)
}
```

---

## Dbałość o sens

### Spójność danych

Nasz generator zachowuje spójność:

1. **Płeć imienia = płeć nazwiska** - Anna Kowalska, nie Anna Kowalski
2. **Kontekst pozostaje** - jeśli oryginalne imię było żeńskie, nowe też będzie
3. **Format zachowany** - jeśli oryginalny telefon miał myślniki, nowy też będzie

### Randomizacja

Randomizacja jest **niezależna od danych wejściowych** - generujemy całkowicie nowe wartości. Jedynie **cechy gramatyczne** (przypadek, rodzaj) są przenoszone dla zachowania poprawności językowej.

Opcjonalnie można ustawić `seed` dla powtarzalności wyników:

```python
generator = SyntheticGenerator(seed=42)
```

---

## Log z przykładami (Showcase)

### Przykład 1: Prosty przypadek - mianownik

**Szablon (zanonimizowany):**
```
Nazywam się {name} {surname}.
```

**Wynik (syntetyczny):**
```
Nazywam się Anna Kowalska.
```

### Przykład 2: Miejscownik - "mieszkam w..."

**Szablon (zanonimizowany):**
```
Mieszkam w {city} na {address}.
```

**Wynik (syntetyczny):**
```
Mieszkam w Krakowie na ul. Długiej 5.
```

✅ **Sukces**: "w Krakowie" - poprawna forma miejscownika

### Przykład 3: Dopełniacz - "jestem z..."

**Szablon (zanonimizowany):**
```
Spotkałem się z {name} z {city}.
```

**Wynik (syntetyczny):**
```
Spotkałem się z Anną z Gdańska.
```

✅ **Sukces**: 
- "z Anną" - narzędnik imienia żeńskiego
- "z Gdańska" - dopełniacz miasta

### Przykład 4: Złożone zdanie z wieloma encjami

**Szablon (zanonimizowany):**
```
{name} {surname} z {city} pracuje w {company}. 
Kontakt: {email}, tel. {phone}.
```

**Wynik (syntetyczny):**
```
Magdalena Wiśniewska z Poznania pracuje w Allegro.
Kontakt: magdalena.wisniewska@gmail.com, tel. +48 512 345 678.
```

### Przykład 5: Trudny przypadek - kontekst emocjonalny

**Szablon (zanonimizowany):**
```
Cześć {name}! Tęsknię za {city}. Pamiętasz jak byliśmy u {name}?
```

**Wynik (syntetyczny):**
```
Cześć Kasiu! Tęsknię za Warszawą. Pamiętasz jak byliśmy u Piotra?
```

✅ **Sukces**:
- "Kasiu" - wołacz (rozpoznany po "Cześć")
- "za Warszawą" - narzędnik (po "tęsknię za")
- "u Piotra" - dopełniacz (po "u")

---

## Ograniczenia

1. **Słowniki są ograniczone** - ~50 unikalnych wartości na typ
2. **Niektóre konstrukcje są trudne** - np. złożone zdania z wielokrotnym użyciem tej samej encji
3. **Brak kontekstu semantycznego** - generator nie rozumie znaczenia zdania

## Możliwe ulepszenia

1. **Integracja z LLM** - użycie modelu językowego do generacji bardziej naturalnych wartości
2. **Większe słowniki** - import z baz danych GUS
3. **Uczenie odmiany** - model seq2seq do automatycznej odmiany dowolnych słów

