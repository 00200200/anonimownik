import re
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

REGEXES = {
    "email": re.compile(r"[a-zA-Z0-9][\w.+-]*@[\w.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"(?:\+?48)?[\s-]?(?:\d{3}[\s-]?){2}\d{3}"),
    "pesel": re.compile(r"\b\d{11}\b"),
    "credit-card-number": re.compile(r"(?:\d[\s-]?){13,19}\b"),
    "bank-account": re.compile(r"(?:\d{2}[\s-]?){13,16}\d{2}\b"),
    "document-number": re.compile(r"\b[A-Z]{2,3}\d{4,7}\b|\b\d{4}-\d{4}-\d{4}\b"),
    "date": re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"),
    "date-of-birth": re.compile(
        r"\b(?:19|20)\d{2}[./-](?:0[1-9]|1[0-2])[./-](?:0[1-9]|[12]\d|3[01])\b"
    ),
}

POLITICAL_WORDS = {
    "pis",
    "platforma obywatelska",
    "po",
    "konfederacja",
    "lewica",
    "prawica",
    "liberał",
    "socjalista",
    "polska 2050",
}


class Anonymizer:
    def __init__(self, model_path: str = "models/herbert-ner"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def regex_hits(self, text: str) -> list[dict]:
        hits = []
        for label, rx in REGEXES.items():
            for m in rx.finditer(text):
                hits.append({"start": m.start(), "end": m.end(), "label": label})
        lower = text.lower()
        for w in POLITICAL_WORDS:
            idx = lower.find(w)
            while idx != -1:
                hits.append(
                    {"start": idx, "end": idx + len(w), "label": "political-view"}
                )
                idx = lower.find(w, idx + 1)
        return hits

    def model_hits(self, text: str) -> list[dict]:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0]
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
        preds = logits.argmax(dim=-1).tolist()
        out = []
        cur = None
        for pred_id, (s, e) in zip(preds, offsets):
            s, e = s.item(), e.item()
            if s == e:
                continue
            label = self.id2label.get(pred_id, "O")
            if label.startswith("B-"):
                if cur:
                    out.append(cur)
                cur = {"start": s, "end": e, "label": label[2:]}
            elif label.startswith("I-") and cur and cur["label"] == label[2:]:
                cur["end"] = e
            else:
                if cur:
                    out.append(cur)
                cur = None
        if cur:
            out.append(cur)
        return out

    def predict(self, text: str) -> list[dict]:
        reg = self.regex_hits(text)
        mod = self.model_hits(text)
        all_ents = reg + mod
        all_ents.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
        merged = []
        for ent in all_ents:
            if merged and not (
                ent["end"] <= merged[-1]["start"] or ent["start"] >= merged[-1]["end"]
            ):
                continue
            merged.append(ent)
        return merged

    def anonymize(self, text: str) -> str:
        ents = self.predict(text)
        if not ents:
            return text
        res = []
        cur = 0
        for ent in sorted(ents, key=lambda x: x["start"]):
            res.append(text[cur : ent["start"]])
            res.append(f"{{{ent['label']}}}")
            cur = ent["end"]
        res.append(text[cur:])
        return "".join(res)


if __name__ == "__main__":
    import sys

    txt = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    print(Anonymizer().anonymize(txt))
