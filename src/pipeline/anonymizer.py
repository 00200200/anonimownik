"""Moduł 1 - Anonimizacja.

Tokenizacja, tagowanie morfosyntaktyczne, NER + reguły.
Wykrywa encje i zapisuje metadane morfologiczne.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import spacy
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Domyślne typy encji do anonimizacji
ENTITY_TYPES = {
    "name", "surname", "age", "date-of-birth", "date", "sex", "religion",
    "political-view", "ethnicity", "sexual-orientation", "health", "relative",
    "city", "address", "email", "phone", "pesel", "document-number",
    "company", "school-name", "job-title", "bank-account", "credit-card-number",
    "username", "secret",
}


@dataclass
class EntityMeta:
    """Metadane wykrytej encji."""
    start: int
    end: int
    entity_type: str
    original_text: str
    morph_features: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "start": self.start,
            "end": self.end,
            "entity_type": self.entity_type,
            "original_text": self.original_text,
            "morph_features": self.morph_features,
        }


@dataclass
class AnonymizationResult:
    """Wynik anonimizacji."""
    anon_text: str
    entities_meta: List[EntityMeta]
    
    def to_dict(self) -> Dict:
        return {
            "anon_text": self.anon_text,
            "entities_meta": [e.to_dict() for e in self.entities_meta],
        }


class Anonymizer:
    """Moduł anonimizacji z wykrywaniem encji i zapisem cech morfologicznych."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        spacy_model: str = "pl_core_news_sm",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Ścieżka do wytrenowanego modelu NER (herbert-ner)
            spacy_model: Nazwa modelu spaCy do analizy morfologicznej
            device: Urządzenie (cuda/cpu), automatyczne jeśli None
        """
        # Załaduj spaCy do analizy morfologicznej
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            raise RuntimeError(
                f"Model spaCy '{spacy_model}' nie jest zainstalowany. "
                f"Uruchom: python -m spacy download {spacy_model}"
            )
        
        # Opcjonalnie załaduj model NER
        self.ner_model = None
        self.ner_tokenizer = None
        self.id2label = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path and Path(model_path).exists():
            self._load_ner_model(model_path)
    
    def _load_ner_model(self, model_path: str):
        """Ładuje model NER."""
        labels_path = Path(model_path) / "labels.json"
        if labels_path.exists():
            labels = json.load(open(labels_path))["label_list"]
            self.id2label = {i: l for i, l in enumerate(labels)}
        
        self.ner_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.ner_model.eval()
        self.ner_model.to(self.device)
    
    def _extract_morph_features(self, text: str, start: int, end: int) -> Dict[str, str]:
        """
        Wyciąga cechy morfologiczne dla fragmentu tekstu.
        
        Args:
            text: Pełny tekst
            start: Początek fragmentu
            end: Koniec fragmentu
        
        Returns:
            Słownik cech morfologicznych (case, gender, number, etc.)
        """
        doc = self.nlp(text)
        
        features = {}
        for token in doc:
            # Sprawdź czy token pokrywa się z naszym fragmentem
            if token.idx >= start and token.idx + len(token.text) <= end:
                morph = token.morph.to_dict()
                if morph:
                    # Zapisz cechy morfologiczne
                    for key in ['Case', 'Gender', 'Number', 'Animacy', 'Aspect']:
                        if key in morph:
                            features[key.lower()] = morph[key]
                break
            elif token.idx <= start < token.idx + len(token.text):
                morph = token.morph.to_dict()
                if morph:
                    for key in ['Case', 'Gender', 'Number', 'Animacy', 'Aspect']:
                        if key in morph:
                            features[key.lower()] = morph[key]
                break
        
        return features
    
    def _predict_ner_spans(self, text: str, threshold: float = 0.3) -> List[Dict]:
        """Wykrywa encje za pomocą modelu NER."""
        if not self.ner_model:
            return []
        
        enc = self.ner_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0]
        enc = {k: v.to(self.device) for k, v in enc.items()}
        
        with torch.no_grad():
            logits = self.ner_model(**enc).logits[0]
        
        probs = torch.softmax(logits, dim=-1)
        pred_ids = probs.argmax(dim=-1)
        pred_conf = probs.max(dim=-1).values
        
        ents = []
        cur = None
        
        for pid, conf, (s, e) in zip(pred_ids, pred_conf, offsets):
            s, e = s.item(), e.item()
            if s == e:
                continue
            
            label = self.id2label[pid.item()]
            
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
    
    def _apply_regex_rules(self, text: str) -> List[Dict]:
        """Wykrywa encje za pomocą reguł regexowych."""
        entities = []
        
        # PESEL - 11 cyfr
        for m in re.finditer(r'\b\d{11}\b', text):
            entities.append({
                "start": m.start(),
                "end": m.end(),
                "label": "pesel",
            })
        
        # Email
        for m in re.finditer(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', text):
            entities.append({
                "start": m.start(),
                "end": m.end(),
                "label": "email",
            })
        
        # Telefon
        for m in re.finditer(r'(?:\+48\s?)?(?:\d{3}[-\s]?){2}\d{3}', text):
            entities.append({
                "start": m.start(),
                "end": m.end(),
                "label": "phone",
            })
        
        # Numer konta bankowego
        for m in re.finditer(r'\b\d{2}(?:\s?\d{4}){6}\b', text):
            entities.append({
                "start": m.start(),
                "end": m.end(),
                "label": "bank-account",
            })
        
        return entities
    
    def predict(self, text: str, use_regex: bool = True) -> List[EntityMeta]:
        """
        Wykrywa encje w tekście.
        
        Args:
            text: Tekst do analizy
            use_regex: Czy używać reguł regexowych
        
        Returns:
            Lista metadanych encji z cechami morfologicznymi
        """
        entities = []
        
        # NER z modelu
        if self.ner_model:
            for ent in self._predict_ner_spans(text):
                if ent["label"] in ENTITY_TYPES:
                    entities.append(ent)
        
        # Reguły regexowe
        if use_regex:
            for ent in self._apply_regex_rules(text):
                # Sprawdź czy nie nachodzi na istniejącą encję
                overlaps = False
                for existing in entities:
                    if not (ent["end"] <= existing["start"] or ent["start"] >= existing["end"]):
                        overlaps = True
                        break
                if not overlaps:
                    entities.append(ent)
        
        # Sortuj po pozycji
        entities.sort(key=lambda e: e["start"])
        
        # Wyciągnij cechy morfologiczne dla każdej encji
        result = []
        for ent in entities:
            original_text = text[ent["start"]:ent["end"]]
            morph_features = self._extract_morph_features(text, ent["start"], ent["end"])
            
            result.append(EntityMeta(
                start=ent["start"],
                end=ent["end"],
                entity_type=ent["label"],
                original_text=original_text,
                morph_features=morph_features,
            ))
        
        return result
    
    def anonymize(self, text: str, use_regex: bool = True) -> AnonymizationResult:
        """
        Anonimizuje tekst, zastępując encje tokenami.
        
        Args:
            text: Tekst do anonimizacji
            use_regex: Czy używać reguł regexowych
        
        Returns:
            AnonymizationResult z anon_text i entities_meta
        """
        entities = self.predict(text, use_regex)
        
        # Zamień encje na tokeny od końca (żeby nie zepsuć indeksów)
        anon_text = text
        for entity in reversed(entities):
            token = f"{{{entity.entity_type}}}"
            anon_text = anon_text[:entity.start] + token + anon_text[entity.end:]
        
        return AnonymizationResult(
            anon_text=anon_text,
            entities_meta=entities,
        )

