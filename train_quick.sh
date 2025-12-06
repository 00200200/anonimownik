#!/bin/bash
# Szybki trening na małej próbce (tylko 100 przykładów) dla testu

echo "==================================================="
echo "QUICK TEST: Trening na 100 przykładach"
echo "==================================================="

# Stwórz małe pliki testowe
head -100 data/nask_train/orig.txt > /tmp/orig_small.txt
head -100 data/nask_train/anonymized.txt > /tmp/anon_small.txt

# Uruchom trening
python src/scripts/train.py \
    --model "allegro/herbert-base-cased" \
    --orig /tmp/orig_small.txt \
    --anonymized /tmp/anon_small.txt \
    --output "models/herbert-deanonymized" \
    --epochs 1 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --save-steps 50

echo ""
echo "==================================================="
echo "Test zakończony!"
echo "==================================================="
