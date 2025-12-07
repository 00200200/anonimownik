import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

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


def load_model(model_path="models/herbert-ner"):
    labels = json.load(open(Path(model_path) / "labels.json"))["label_list"]
    id2label = {i: l for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForTokenClassification.from_pretrained(model_path)
    mdl.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    return tok, mdl, id2label, device


def predict_spans(text, tokenizer, model, id2label, device, threshold=0.5):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")[0]
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1)
    pred_ids = probs.argmax(dim=-1)
    pred_conf = probs.max(dim=-1).values
    ents = []
    cur = None
    for pid, conf, (s, e) in zip(pred_ids, pred_conf, offsets):
        s, e = s.item(), e.item()
        if s == e:
            continue
        label = id2label[pid.item()]
        if label == "O" or conf.item() < threshold:
            if cur:
                ents.append(cur)
                cur = None
            continue
        if label.startswith("B-"):
            if cur:
                ents.append(cur)
            cur = {"start": s, "end": e, "label": label[2:]}
        elif label.startswith("I-") and cur and cur["label"] == label[2:]:
            cur["end"] = e
        else:
            if cur:
                ents.append(cur)
            cur = None
    if cur:
        ents.append(cur)
    return ents


def score(threshold, tokenizer, model, id2label, device, test_path):
    tp = fp = fn = 0
    for line in open(test_path):
        data = json.loads(line)
        true_spans = {
            (e["start"], e["end"], e["label"])
            for e in data["entities"]
            if e["label"] in OFFICIAL_LABELS
        }
        pred_spans = {
            (e["start"], e["end"], e["label"])
            for e in predict_spans(
                data["text"], tokenizer, model, id2label, device, threshold
            )
            if e["label"] in OFFICIAL_LABELS
        }
        tp += len(true_spans & pred_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return p, r, f1, tp, fp, fn


def evaluate(test_path: str = "data/processed/test.json"):
    tok, mdl, id2label, device = load_model()
    th = 0.3
    p, r, f1, tp, fp, fn = score(th, tok, mdl, id2label, device, test_path)
    print(f"th={th:.1f} P={p:.4f} R={r:.4f} F1={f1:.4f} TP={tp} FP={fp} FN={fn}")


if __name__ == "__main__":
    evaluate()
