#!/bin/bash
# Skrypt do uruchomienia treningu modelu Herbert z PyTorch Lightning

echo "==================================================="
echo "TRENING MODELU HERBERT (PyTorch Lightning)"
echo "==================================================="

# Sprawdź czy istnieją pliki danych
if [ ! -f "data/nask_train/orig.txt" ]; then
    echo "BŁĄD: Nie znaleziono pliku data/nask_train/orig.txt"
    exit 1
fi

if [ ! -f "data/nask_train/anonymized.txt" ]; then
    echo "BŁĄD: Nie znaleziono pliku data/nask_train/anonymized.txt"
    exit 1
fi

# Domyślne parametry
MODEL=${1:-"allegro/herbert-base-cased"}
EPOCHS=${2:-3}
BATCH_SIZE=${3:-1}
ACCUMULATE=${4:-8}
OUTPUT_DIR="models/herbert-deanonymized"

echo ""
echo "Parametry treningu:"
echo "  - Model bazowy: $MODEL"
echo "  - Epochs: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Gradient accumulation: $ACCUMULATE"
echo "  - Efektywny batch size: $((BATCH_SIZE * ACCUMULATE))"
echo "  - Output: $OUTPUT_DIR"
echo ""

# Uruchom trening
python src/scripts/train.py \
    --model "$MODEL" \
    --orig data/nask_train/orig.txt \
    --anonymized data/nask_train/anonymized.txt \
    --output "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate 5e-5 \
    --accumulate "$ACCUMULATE"

echo ""
echo "==================================================="
echo "Trening zakończony!"
echo "Aby użyć wytrenowanego modelu:"
echo "python src/scripts/deanonymize_herbert.py input.txt output.txt --model $OUTPUT_DIR"
echo "==================================================="
