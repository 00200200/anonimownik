import lightning as L
import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torchmetrics import F1Score
from typing import List, Dict


class NERModel(L.LightningModule):
    def __init__(
        self,
        model_name: str = "allegro/herbert-base-cased",
        num_labels: int = 25,
        lr: float = 2e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # F1 metric
        self.train_f1 = F1Score(
            task="multiclass", num_classes=num_labels, average="macro"
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_labels, average="macro"
        )

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(
            outputs.logits.view(-1, self.hparams.num_labels),
            labels.view(-1),
            ignore_index=-100,
        )

        # F1
        preds = torch.argmax(outputs.logits, dim=-1)
        mask = labels != -100
        if mask.any():
            self.train_f1(preds[mask], labels[mask])

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_f1", self.train_f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(
            outputs.logits.view(-1, self.hparams.num_labels),
            labels.view(-1),
            ignore_index=-100,
        )

        # F1
        preds = torch.argmax(outputs.logits, dim=-1)
        mask = labels != -100
        if mask.any():
            self.val_f1(preds[mask], labels[mask])

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def predict_entities(self, text: str, label_map: Dict[int, str]) -> List[Dict]:
        """Przewiduje entities w tekście"""
        self.eval()

        # Tokenizacja
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        offset_mapping = encoding.pop("offset_mapping")[0]
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Predykcja
        with torch.no_grad():
            outputs = self(input_ids, attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]

        # Mapuj tokeny na entities
        entities = []
        current_entity = None

        for idx, (pred_id, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == end:  # Special token
                continue

            pred_label = label_map.get(pred_id.item(), "O")

            if pred_label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "start": start.item(),
                    "end": end.item(),
                    "label": pred_label[2:],
                }
            elif pred_label.startswith("I-") and current_entity:
                current_entity["end"] = end.item()
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities
