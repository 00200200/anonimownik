#!/usr/bin/env python3
"""
Skrypt do fine-tuningu modelu Herbert na danych z tokenami.
Trenuje model na parach: tekst z tokenami (orig.txt) → tekst deanonimizowany (anonymized.txt)
Uwzględnia rodzaj tokenu podczas treningu poprzez dodanie specjalnych markerów typu.
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from datasets import Dataset as HFDataset


class DeanonymizationDataset(Dataset):
    """Dataset do trenowania modelu na zadaniu deanonimizacji z uwzględnieniem typu tokenu."""
    
    def __init__(self, tokenized_texts, anonymized_texts, tokenizer, max_length=512):
        self.tokenized_texts = tokenized_texts
        self.anonymized_texts = anonymized_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._prepare_examples()
    
    def _prepare_examples(self) -> List[dict]:
        """Przygotuj przykłady treningowe - zamień tokeny na [MASK] wraz z typem tokenu."""
        examples = []
        
        token_pattern = r'\[(name|surname|city|address|phone|email|pesel|age|sex|document-number|company|date|data|bank-account)\]'
        
        for tokenized, anonymized in zip(self.tokenized_texts, self.anonymized_texts):
            # Znajdź wszystkie tokeny w tekście
            matches = list(re.finditer(token_pattern, tokenized, re.IGNORECASE))
            
            if not matches:
                continue
            
            # Dla każdego tokenu stwórz przykład treningowy
            for match in matches:
                token_type = match.group(1).lower()  # Pobierz typ tokenu (name, pesel, etc.)
                
                # Pobierz kontekst wokół tokenu
                start_ctx = max(0, match.start() - 200)
                end_ctx = min(len(tokenized), match.end() + 200)
                
                # Tekst z [MASK] + informacją o typie tokenu
                # Format: "[MASK:type]" np. "[MASK:name]" lub "[MASK:pesel]"
                masked_text = (
                    tokenized[start_ctx:match.start()] + 
                    f"[MASK:{token_type}]" + 
                    tokenized[match.end():end_ctx]
                )
                
                # Znajdź odpowiadającą wartość w tekście deanonimizowanym
                # To jest uproszczenie - zakładamy że kolejność jest taka sama
                target_text = anonymized[start_ctx:end_ctx]
                
                examples.append({
                    'input_text': masked_text,
                    'target_text': target_text,
                    'token_type': token_type  # Zachowaj typ tokenu
                })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenizuj input z maską i typem
        encoding = self.tokenizer(
            example['input_text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Usuń wymiar batch
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


def load_data(orig_path: str, anonymized_path: str) -> Tuple[List[str], List[str]]:
    """Wczytaj dane z plików."""
    print(f"Ładowanie danych z {orig_path} i {anonymized_path}...")
    
    with open(orig_path, 'r', encoding='utf-8') as f:
        orig_text = f.read()
    
    with open(anonymized_path, 'r', encoding='utf-8') as f:
        anon_text = f.read()
    
    # Podziel na akapity/zdania dla lepszego treningu
    orig_paragraphs = [p.strip() for p in orig_text.split('\n') if p.strip()]
    anon_paragraphs = [p.strip() for p in anon_text.split('\n') if p.strip()]
    
    print(f"Załadowano {len(orig_paragraphs)} akapitów z orig.txt")
    print(f"Załadowano {len(anon_paragraphs)} akapitów z anonymized.txt")
    
    # Upewnij się że mamy tyle samo
    min_len = min(len(orig_paragraphs), len(anon_paragraphs))
    return orig_paragraphs[:min_len], anon_paragraphs[:min_len]


def prepare_typed_mlm_dataset(orig_texts: List[str], anon_texts: List[str], tokenizer, sample_size: int = None) -> HFDataset:
    """
    Przygotuj dataset który uwzględnia typ tokenu.
    Dla każdej linii znajduje tokeny [type] i tworzy warianty z [MASK:type].
    """
    
    print("Przygotowywanie datasetu z uwzględnieniem typów tokenów...")
    
    token_pattern = r'\[(name|surname|city|address|phone|email|pesel|age|sex|document-number|company|date|data|bank-account)\]'
    
    all_texts = []
    
    for anon_text, orig_text in zip(anon_texts, orig_texts):
        # Znajdź wszystkie tokeny
        matches = list(re.finditer(token_pattern, anon_text, re.IGNORECASE))
        
        if matches:
            # Utwórz wariant z tokenami zamienionymi na [MASK:type]
            text_with_typed_masks = anon_text
            # Zamieniaj od końca, żeby nie zepsuć indeksów
            for match in reversed(matches):
                token_type = match.group(1).lower()
                text_with_typed_masks = (
                    text_with_typed_masks[:match.start()] + 
                    f"[MASK:{token_type}]" + 
                    text_with_typed_masks[match.end():]
                )
            all_texts.append(text_with_typed_masks)
        else:
            # Jeśli nie ma tokenów, użyj oryginalnego tekstu
            all_texts.append(orig_text)
    
    # Jeśli podano sample_size, użyj tylko części danych (dla małej pamięci)
    if sample_size:
        all_texts = all_texts[:sample_size]
        print(f"Używam {sample_size} próbek (z {len(all_texts)} dostępnych)")
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=256)
    
    # Stwórz dataset HuggingFace
    dataset_dict = {'text': all_texts}
    dataset = HFDataset.from_dict(dataset_dict)
    
    # Tokenizuj
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    print(f"Dataset przygotowany: {len(tokenized_dataset)} przykładów")
    print(f"Przykład tekstu z typowanymi maskami: {all_texts[0][:200]}...")
    
    return tokenized_dataset


def train_model(
    model_name: str,
    orig_path: str,
    anonymized_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    save_steps: int = 500
):
    """Trenuj model Herbert na danych deanonimizacji."""
    
    print("="*60)
    print("FINE-TUNING MODELU HERBERT NA DEANONIMIZACJI")
    print("="*60)
    
    # Sprawdź czy katalog wyjściowy jest zapisywalny
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    test_file = output_path / ".write_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        print(f"✅ Katalog {output_dir} jest zapisywalny\n")
    except Exception as e:
        print(f"❌ BŁĄD: Nie można zapisywać do {output_dir}: {e}")
        raise
    
    # Wczytaj model i tokenizer
    print(f"\nŁadowanie modelu bazowego: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Dodaj nowe tokeny typu [MASK:name], [MASK:pesel] itp. do vocabulary
    print("\nDodawanie specjalnych tokenów typu do vocabulary...")
    entity_types = [
        'name', 'surname', 'city', 'address', 'phone', 'email', 'pesel', 
        'age', 'sex', 'document-number', 'company', 'date', 'data', 'bank-account'
    ]
    new_tokens = [f"[MASK:{etype}]" for etype in entity_types]
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"✅ Dodano {num_added} nowych tokenów do vocabulary")
    
    # Wczytaj model i rozszerz embedding layer
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    print(f"✅ Rozszerzono embedding layer do {len(tokenizer)} tokenów")
    
    # Wczytaj dane
    orig_texts, anon_texts = load_data(orig_path, anonymized_path)
    
    # Przygotuj dataset (użyj wszystkich danych) z uwzględnieniem typów tokenów
    train_dataset = prepare_typed_mlm_dataset(orig_texts, anon_texts, tokenizer, sample_size=None)
    
    # Data collator dla MLM (automatycznie maskuje tokeny)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # 15% tokenów będzie maskowanych
    )
    
    # Argumenty treningu - zoptymalizowane pod małą ilość pamięci
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,  # Symuluje batch_size * 8
        save_steps=save_steps,
        save_total_limit=2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        logging_dir=f"{output_dir}/logs",
        report_to="none",  # Wyłącz wandb/tensorboard
        fp16=torch.cuda.is_available(),  # Użyj mixed precision jeśli GPU
        dataloader_num_workers=0,  # Zmniejsz użycie pamięci przez dataloader
        max_grad_norm=1.0,  # Gradient clipping
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Trenuj
    print("\nRozpoczynanie treningu...")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Training samples: {len(train_dataset)}")
    
    try:
        trainer.train()
        
        # Zapisz model
        print(f"\nZapisywanie wytrenowanego modelu do {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Weryfikacja zapisu
        saved_files = list(Path(output_dir).glob('*'))
        print(f"\n✅ Zapisano {len(saved_files)} plików:")
        for f in saved_files[:10]:  # Pokaż pierwsze 10
            print(f"   - {f.name}")
        if len(saved_files) > 10:
            print(f"   ... i {len(saved_files) - 10} więcej")
        
        print("\n" + "="*60)
        print("TRENING ZAKOŃCZONY!")
        print(f"Model zapisany w: {output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ BŁĄD podczas treningu: {e}")
        print("\nSzczegóły błędu:")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tuning modelu Herbert na danych deanonimizacji'
    )
    parser.add_argument(
        '--model',
        default='allegro/herbert-base-cased',
        help='Nazwa bazowego modelu Herbert'
    )
    parser.add_argument(
        '--orig',
        default='data/nask_train/orig.txt',
        help='Plik z tokenami (orig.txt)'
    )
    parser.add_argument(
        '--anonymized',
        default='data/nask_train/anonymized.txt',
        help='Plik z danymi deanonimizowanymi (anonymized.txt)'
    )
    parser.add_argument(
        '--output',
        default='models/herbert-deanonymized',
        help='Katalog do zapisania wytrenowanego modelu'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Liczba epok treningu'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--save-steps',
        type=int,
        default=500,
        help='Zapisuj checkpoint co N kroków'
    )
    
    args = parser.parse_args()
    
    # Utwórz katalog wyjściowy
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Trenuj
    train_model(
        model_name=args.model,
        orig_path=args.orig,
        anonymized_path=args.anonymized,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps
    )


if __name__ == '__main__':
    main()