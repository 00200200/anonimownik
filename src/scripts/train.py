import json
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_scheduler

OFFICIAL_LABELS = {
    "name",
    "surname",
    "age",
    "date-of-birth",
    "date",
    "sex",
    "religion",
    "political-view",
    "ethnicity",
    "sexual-orientation",
    "health",
    "relative",
    "city",
    "address",
    "email",
    "phone",
    "pesel",
    "document-number",
    "company",
    "school-name",
    "job-title",
    "bank-account",
    "credit-card-number",
    "username",
    "secret",
}


def build_label_list() -> list[str]:
    """BIO tags for official labels only."""
    labels = ["O"]
    for label in sorted(OFFICIAL_LABELS):
        labels.extend([f"B-{label}", f"I-{label}"])
    return labels


LABEL_LIST = build_label_list()
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


class NERDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        for line in open(path):
            data = json.loads(line)
            entities = [
                e for e in data.get("entities", []) if e["label"] in OFFICIAL_LABELS
            ]
            if entities:
                self.samples.append({"text": data["text"], "entities": entities})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        enc = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_offsets_mapping=True,
        )

        labels = []
        for s, e in enc["offset_mapping"]:
            if s == e:
                labels.append(-100)
                continue
            tag = "O"
            for ent in sample["entities"]:
                if s >= ent["start"] and e <= ent["end"]:
                    prefix = "B-" if s == ent["start"] else "I-"
                    tag = prefix + ent["label"]
                    break
            labels.append(LABEL2ID.get(tag, 0))

        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(labels),
        }


class NERModel(L.LightningModule):
    def __init__(
        self, model_name: str = "allegro/herbert-base-cased", lr: float = 2e-5
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID
        )
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        self.log("train_loss", out.loss, prog_bar=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        self.log("val_loss", out.loss, prog_bar=True)
        return out.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = get_scheduler(
            "cosine",
            optimizer,
            num_warmup_steps=100,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def train(
    train_path: str = "data/processed/train.json",
    val_path: str = "data/processed/test.json",
    warmup_path: str = "data/processed/train_synth.json",
    output_dir: str = "models/herbert-ner",
    model_name: str = "allegro/herbert-base-cased",
    epochs: int = 6,
    warmup_epochs: int = 1,
    batch_size: int = 8,
    lr: float = 3e-5,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = NERDataset(train_path, tokenizer)
    val_ds = NERDataset(val_path, tokenizer)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    print(f"Labels: {len(LABEL_LIST)} ({len(OFFICIAL_LABELS)} categories)")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    model = NERModel(model_name, lr)

    if Path(warmup_path).exists():
        warmup_ds = NERDataset(warmup_path, tokenizer)
        warmup_loader = DataLoader(
            warmup_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )
        trainer_warmup = L.Trainer(
            max_epochs=warmup_epochs,
            accelerator="auto",
            devices=1,
            precision="16-mixed" if torch.cuda.is_available() else 32,
            log_every_n_steps=20,
            enable_checkpointing=False,
            logger=False,
        )
        trainer_warmup.fit(model, warmup_loader)

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        log_every_n_steps=20,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(model, train_loader, val_loader)

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(Path(output_dir) / "labels.json", "w") as f:
        json.dump({"label_list": LABEL_LIST}, f, ensure_ascii=False, indent=2)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    train()
