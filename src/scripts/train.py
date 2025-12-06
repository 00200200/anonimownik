"""Trening modelu Herbert z użyciem PyTorch Lightning."""

import argparse
import sys
from pathlib import Path

# Dodaj główny katalog projektu do PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling

from src.data.dataset import DeanonymizationDataset, load_data


class HerbertMLM(pl.LightningModule):
    """Lightning module do treningu Herbert MLM."""
    
    def __init__(
        self,
        model_name: str = "allegro/herbert-base-cased",
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Załaduj model
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Data collator (automatycznie maskuje tokeny)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
    def save_model(self, output_dir: str):
        """Zapisz model i tokenizer."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nZapisywanie modelu do {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Weryfikacja
        saved_files = list(output_path.glob("*"))
        print(f"✅ Zapisano {len(saved_files)} plików:")
        for f in sorted(saved_files)[:10]:
            print(f"   - {f.name}")
        if len(saved_files) > 10:
            print(f"   ... i {len(saved_files) - 10} więcej")


def train(
    model_name: str,
    orig_path: str,
    anon_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 5e-5,
    max_length: int = 256,
    accumulate_grad_batches: int = 8,
    num_workers: int = 15
):
    """Trenuj model Herbert."""
    print("=" * 60)
    print("=" * 60)
    
    # Załaduj dane
    orig_texts, anon_texts = load_data(orig_path, anon_path)
    
    # Stwórz model
    print(f"Inicjalizacja modelu: {model_name}")
    pl_model = HerbertMLM(
        model_name=model_name,
        learning_rate=learning_rate
    )
    
    # Stwórz dataset
    dataset = DeanonymizationDataset(
        orig_texts=orig_texts,
        anon_texts=anon_texts,
        tokenizer=pl_model.tokenizer,
        max_length=max_length
    )
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pl_model.data_collator
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="herbert-{epoch:02d}-{train_loss:.2f}",
        save_top_k=2,
        monitor="train_loss",
        mode="min"
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=50,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
    )
    
    # Trenuj
    print(f"\nRozpoczynam trening...")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Gradient accumulation: {accumulate_grad_batches}")
    print(f"  - Efektywny batch size: {batch_size * accumulate_grad_batches}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Samples: {len(dataset)}")
    print()
    
    try:
        trainer.fit(pl_model, dataloader)
        
        # Zapisz finalny model
        pl_model.save_model(output_dir)
        
        print("\n" + "=" * 60)
        print("✅ TRENING ZAKOŃCZONY!")
        print(f"Model zapisany w: {output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ BŁĄD podczas treningu: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="Trenuj model Herbert na danych deanonimizacji")
    
    parser.add_argument("--model", default="allegro/herbert-base-cased", help="Nazwa modelu bazowego")
    parser.add_argument("--orig", required=True, help="Plik z tokenami (orig.txt)")
    parser.add_argument("--anonymized", required=True, help="Plik z danymi deanonimizowanymi")
    parser.add_argument("--output", required=True, help="Katalog do zapisania modelu")
    parser.add_argument("--epochs", type=int, default=3, help="Liczba epok")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256, help="Maksymalna długość sekwencji")
    parser.add_argument("--accumulate", type=int, default=8, help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    train(
        model_name=args.model,
        orig_path=args.orig,
        anon_path=args.anonymized,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        accumulate_grad_batches=args.accumulate
    )


if __name__ == "__main__":
    main()
