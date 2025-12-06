# Quick Start - Deanonimizacja

## Szybkie użycie (bez treningu)

Możesz używać modelu od razu bez trenowania:

```bash
# Test na małym przykładzie
python test_deanonymize.py

# Przetwórz plik
python src/scripts/deanonymize_herbert.py input.txt output.txt
```

Model użyje bazowego `allegro/herbert-base-cased`, który już zna polski język.

## Z wytrenowanym modelem

Jeśli chcesz lepszych wyników, wytrenuj model na swoich danych:

```bash
# 1. Wytrenuj model
./train.sh

# 2. Użyj wytrenowanego modelu
python src/scripts/deanonymize_herbert.py input.txt output.txt --model models/herbert-deanonymized
```

## Format pliku wejściowego

Plik powinien zawierać tokeny w nawiasach kwadratowych:

```
Nazywam się [name] [surname], mam [age] lat.
Mieszkam w [city] przy [address].
Telefon: [phone], email: [email]
```

## Dostępne tokeny

- `[name]` - imię
- `[surname]` - nazwisko
- `[city]` - miasto
- `[address]` - adres
- `[company]` - nazwa firmy
- `[phone]` - numer telefonu
- `[email]` - adres email
- `[age]` - wiek
- `[sex]` - płeć
- `[pesel]` - numer PESEL
- `[document-number]` - numer dokumentu
- `[date]` - data
- `[bank-account]` - numer konta

## Troubleshooting

### Model nie znaleziony

Jeśli widzisz ostrzeżenie o braku modelu, to normalne - użyje bazowego modelu Herbert.

Aby wytrenować własny model:
```bash
./train.sh
```

### Brak pliku input.txt

Utwórz plik z tokenami:
```bash
echo "Nazywam się [name] [surname], mam [age] lat." > input.txt
python src/scripts/deanonymize_herbert.py input.txt output.txt
```
