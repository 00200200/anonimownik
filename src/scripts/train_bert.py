import json
import lightning as L
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
from src.model.model_encrypter import NERModel
from collections import Counter


def get_labels_from_data(filepath: str):
    labels_set = set()
    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line)
            for ent in data["entities"]:
                labels_set.add(ent["label"])

    labels = ["O"]
    for label in sorted(labels_set):
        labels.append(f"B-{label}")
        labels.append(f"I-{label}")

    return labels


LABELS = get_labels_from_data("data/processed/train.json")
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


class NERDataset(Dataset):
    def __init__(self, filepath: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line)
                if data["entities"]:
                    self.examples.append(data)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example["text"]
        entities = example["entities"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = encoding.pop("offset_mapping")

        labels = []
        for i, (start, end) in enumerate(offset_mapping):
            if start == end:
                labels.append(-100)
                continue

            token_label = "O"
            for entity in entities:
                e_start, e_end = entity["start"], entity["end"]

                if start >= e_start and end <= e_end:
                    prefix = "B-" if start == e_start else "I-"
                    token_label = prefix + entity["label"]
                    break

            labels.append(LABEL2ID.get(token_label, 0))

        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(labels),
        }


def train():
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")

    train_dataset = NERDataset("data/processed/train.json", tokenizer)
    val_dataset = NERDataset("data/processed/test.json", tokenizer)

    print(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Labels: {len(LABELS)}"
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4)

    model = NERModel(
        model_name="allegro/herbert-base-cased", num_labels=len(LABELS), lr=2e-5
    )

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=10,
        val_check_interval=0.5,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("models/ner_model.ckpt")


if __name__ == "__main__":
    train()
