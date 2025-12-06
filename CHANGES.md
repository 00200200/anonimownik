# Zmiany w logice treningu - Uwzględnienie typu tokenu

## Co zostało zmienione?

### 1. **Nowe tokeny typu w vocabulary** (`train.py`)
Model Herbert został rozszerzony o specjalne tokeny, które zawierają informację o typie danych:
- `[MASK:name]` - dla imion
- `[MASK:surname]` - dla nazwisk
- `[MASK:pesel]` - dla numerów PESEL
- `[MASK:city]` - dla miast
- `[MASK:address]` - dla adresów
- `[MASK:phone]` - dla telefonów
- `[MASK:email]` - dla emaili
- ...i wszystkie inne typy

### 2. **Nowa funkcja trenowania** (`prepare_typed_mlm_dataset`)
Zamiast zamieniać wszystkie tokeny na zwykłe `[MASK]`, teraz:
```python
[name] → [MASK:name]
[pesel] → [MASK:pesel]
[city] → [MASK:city]
```

Model podczas treningu **wie**, jakiego typu dane ma przewidywać.

### 3. **Rozszerzenie embedding layer**
```python
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))
```
Model ma teraz większy vocabulary, który zawiera wszystkie typy tokenów.

### 4. **Kompatybilność wsteczna**
Wszystkie skrypty deanonimizacji (`deanonymize.py`, `deanonymize_herbert.py`) obsługują **oba formaty**:
- Stary format: `[name]`, `[pesel]`
- Nowy format: `[MASK:name]`, `[MASK:pesel]`

## Korzyści

### Przed zmianą:
```
Tekst: "Nazywam się [name] i mój PESEL to [pesel]"
       ↓
Trening: "Nazywam się [MASK] i mój PESEL to [MASK]"
```
❌ Model nie wie, czy pierwszy `[MASK]` to imię czy PESEL

### Po zmianie:
```
Tekst: "Nazywam się [name] i mój PESEL to [pesel]"
       ↓
Trening: "Nazywam się [MASK:name] i mój PESEL to [MASK:pesel]"
```
✅ Model wie dokładnie, jakiego typu dane ma przewidywać

## Jak użyć?

### Trening modelu:
```bash
python src/scripts/train.py \
    --orig data/nask_train/orig.txt \
    --anonymized data/nask_train/anonymized.txt \
    --output models/herbert-typed \
    --epochs 5
```

Model automatycznie:
1. Doda nowe tokeny do vocabulary
2. Rozszerzy embedding layer
3. Będzie trenował z uwzględnieniem typów

### Deanonimizacja:
```bash
# Stary format tokenów nadal działa
python src/scripts/deanonymize.py input.txt output.txt

# Nowy model z typowanymi tokenami
python src/scripts/deanonymize_herbert.py \
    --model models/herbert-typed \
    input.txt output.txt
```

## Uwagi techniczne

1. **Rozmiar modelu**: Zwiększył się nieznacznie (+14 tokenów w vocabulary)
2. **Czas treningu**: Bez zmian
3. **Dokładność**: Powinna się znacząco poprawić, ponieważ model ma więcej informacji
4. **Kompatybilność**: Pełna kompatybilność wsteczna z istniejącymi danymi

## Pliki zmodyfikowane:
- `src/scripts/train.py` - główna logika treningu
- `src/scripts/deanonymize.py` - obsługa nowego formatu
- `src/scripts/deanonymize_herbert.py` - obsługa nowego formatu

