import json
from collections import defaultdict
from anonymizer import Anonymizer

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


def evaluate(test_path: str = "data/processed/test.json"):
    anon = Anonymizer()
    tp = fp = fn = 0
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for line in open(test_path):
        data = json.loads(line)
        true_spans = {
            (e["start"], e["end"], e["label"])
            for e in data["entities"]
            if e["label"] in OFFICIAL_LABELS
        }
        pred_spans = {
            (e["start"], e["end"], e["label"])
            for e in anon.predict(data["text"])
            if e["label"] in OFFICIAL_LABELS
        }

        for s, e, label in true_spans & pred_spans:
            stats[label]["tp"] += 1
        for s, e, label in pred_spans - true_spans:
            stats[label]["fp"] += 1
        for s, e, label in true_spans - pred_spans:
            stats[label]["fn"] += 1

        tp += len(true_spans & pred_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)

    print("=== Per-label F1 ===")
    for label in sorted(OFFICIAL_LABELS):
        s = stats[label]
        total = s["tp"] + s["fn"]
        if total == 0:
            continue
        lp = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0
        lr = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0
        lf1 = 2 * lp * lr / (lp + lr) if lp + lr else 0
        print(f"{label:20s} F1={lf1:.3f}  P={lp:.3f} R={lr:.3f}  (n={total})")

    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0

    print(f"\n=== Overall ===")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")


if __name__ == "__main__":
    evaluate()
