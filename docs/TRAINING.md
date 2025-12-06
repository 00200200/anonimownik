# Przewodnik Fine-tuningu

## 1. Przygotowanie środowiska

```bash
# Zainstaluj zależności (jeśli jeszcze nie)
uv sync

# Lub pip
pip install accelerate datasets torch transformers
```

## 2. Sprawdź dane

Upewnij się, że masz przygotowane pliki:

```bash
ls -lh data/nask_train/orig.txt
ls -lh data/nask_train/anonymized.txt
```

Pliki powinny mieć tę samą liczbę linii i odpowiadające sobie treści.

## 3. Uruchom trening

### Opcja A: Prosty skrypt (zalecane)

```bash
./train.sh
```

### Opcja B: Z niestandardowymi parametrami

```bash
# Mniej epok, mniejszy batch (dla słabszego sprzętu)
./train.sh "allegro/herbert-base-cased" 2 2

# Więcej epok dla lepszych rezultatów
./train.sh "allegro/herbert-base-cased" 5 4
```

### Opcja C: Pełna kontrola

```bash
python src/scripts/train.py \
    --model allegro/herbert-base-cased \
    --orig data/nask_train/orig.txt \
    --anonymized data/nask_train/anonymized.txt \
    --output models/herbert-finetuned \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 5e-5 \
    --save-steps 500
```

## 4. Monitorowanie treningu

Trening wyświetla postęp:
```
Step 100/1500: loss=2.345
Step 200/1500: loss=1.987
...
```

## 5. Użycie wytrenowanego modelu

Po zakończeniu treningu użyj modelu:

```bash
python src/scripts/deanonymize_herbert.py \
    data/nask_train/orig.txt \
    output/deanonymized.txt \
    --model models/herbert-deanonymized
```

## Rozwiązywanie problemów

### Brak pamięci (CUDA out of memory)

Zmniejsz batch_size:
```bash
./train.sh "allegro/herbert-base-cased" 3 2
```

### Trening na CPU (brak GPU)

Będzie wolniejszy, ale zadziała. System automatycznie wykryje i użyje CPU.

### Zbyt długi trening

Zmniejsz liczbę epok:
```bash
./train.sh "allegro/herbert-base-cased" 1 4
```

## Porównanie wydajności

- **CPU**: ~2-3h dla 3 epok
- **GPU (GTX 1080)**: ~30-45 min dla 3 epok  
- **GPU (RTX 3090)**: ~15-20 min dla 3 epok

## Jakość modelu

Po treningu model powinien:
- Lepiej przewidywać polskie imiona i nazwiska
- Generować bardziej kontekstowe miasta
- Rozumieć strukturę dokumentów w zbiorze

Przetestuj jakość:
```bash
python test_deanonymize.py
```
