"""Regex-based masking dla prostych kategorii danych"""

import re
from typing import List, Dict, Tuple


# Wzorce regex
PATTERNS = {
    "pesel": r"\b\d{11}\b",
    "phone": r"(?:\+48[\s-]?)?(?:(?:\d{3}[\s-]?\d{3}[\s-]?\d{3})|(?:\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}))\b",
    "email": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
    "bank-account": r"\b\d{26}\b",
    "credit-card-number": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
}


def compile_patterns():
    """Kompiluje wzorce regex dla wydajności"""
    return {label: re.compile(pattern) for label, pattern in PATTERNS.items()}


COMPILED = compile_patterns()


def find_entities(text: str, labels: List[str] = None) -> List[Dict]:
    """
    Znajduje entities w tekście używając regexów.

    Args:
        text: Tekst do przeszukania
        labels: Lista labelek do wykrycia (None = wszystkie)

    Returns:
        Lista entities w formacie: {"start": int, "end": int, "label": str}
    """
    if labels is None:
        labels = list(COMPILED.keys())

    entities = []

    # Najpierw wykryj PESEL (priorytet)
    if "pesel" in labels and "pesel" in COMPILED:
        pattern = COMPILED["pesel"]
        for match in pattern.finditer(text):
            entities.append(
                {"start": match.start(), "end": match.end(), "label": "pesel"}
            )

    # Potem inne labelki
    pesel_spans = {(e["start"], e["end"]) for e in entities if e["label"] == "pesel"}

    for label in labels:
        if label == "pesel" or label not in COMPILED:
            continue

        pattern = COMPILED[label]
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()

            # Sprawdź czy nie nachodzi na PESEL
            overlaps = False
            for pesel_start, pesel_end in pesel_spans:
                if not (end <= pesel_start or start >= pesel_end):
                    overlaps = True
                    break

            if not overlaps:
                entities.append({"start": start, "end": end, "label": label})

    # Sortuj według pozycji
    entities.sort(key=lambda e: e["start"])
    return entities


def mask_text(text: str, labels: List[str] = None) -> Tuple[str, List[Dict]]:
    """
    Maskuje tekst zamieniając wykryte entities na tokeny.

    Args:
        text: Tekst do zamaskowania
        labels: Lista labelek do wykrycia (None = wszystkie)

    Returns:
        Tuple (zamaskowany_tekst, lista_entities)
    """
    entities = find_entities(text, labels)

    if not entities:
        return text, []

    # Buduj zamaskowany tekst od końca żeby indeksy się nie zmieniały
    masked = text
    for entity in reversed(entities):
        start = entity["start"]
        end = entity["end"]
        label = entity["label"]
        masked = masked[:start] + f"[{label}]" + masked[end:]

    return masked, entities


def unmask_text(masked_text: str, entities: List[Dict], original_text: str) -> str:
    """
    Odwraca maskowanie - zamienia tokeny z powrotem na oryginalne wartości.

    Args:
        masked_text: Tekst z tokenami
        entities: Lista entities z pozycjami w ORYGINALNYM tekście
        original_text: Oryginalny tekst

    Returns:
        Tekst z przywróconymi wartościami
    """
    result = masked_text

    # Dla każdej entity zamień token na oryginalną wartość
    for entity in entities:
        original_value = original_text[entity["start"] : entity["end"]]
        token = f"[{entity['label']}]"
        result = result.replace(token, original_value, 1)

    return result


def apply_to_dataset(dataset: List[Dict], labels: List[str] = None) -> List[Dict]:
    """
    Aplikuje maskowanie regex na cały dataset.
    Dodaje nowe entities wykryte regexem do istniejących.

    Args:
        dataset: Lista przykładów z train.json/test.json
        labels: Lista labelek do wykrycia

    Returns:
        Dataset z dodanymi entities z regex
    """
    result = []

    for item in dataset:
        text = item["text"]
        existing_entities = item.get("entities", [])

        # Znajdź nowe entities regexem
        new_entities = find_entities(text, labels)

        # Połącz z istniejącymi (usuń duplikaty na podstawie pozycji)
        all_entities = existing_entities.copy()
        existing_spans = {(e["start"], e["end"]) for e in existing_entities}

        for entity in new_entities:
            span = (entity["start"], entity["end"])
            if span not in existing_spans:
                all_entities.append(entity)

        # Sortuj według pozycji
        all_entities.sort(key=lambda e: e["start"])

        result.append({**item, "entities": all_entities})

    return result


if __name__ == "__main__":
    # Test
    test_text = (
        "Mój PESEL to 90010112345 a telefon 501 222 333. Email: test@example.com"
    )

    print("Tekst oryginalny:")
    print(test_text)

    print("\nWykryte entities:")
    entities = find_entities(test_text)
    for ent in entities:
        value = test_text[ent["start"] : ent["end"]]
        print(f"  {ent['label']}: {value}")

    print("\nTekst zamaskowany:")
    masked, _ = mask_text(test_text)
    print(masked)

    print("\nTest z datasetem:")
    sample_dataset = [
        {
            "text": "Nazywam się Jan Kowalski, PESEL 12345678901, tel 600123456",
            "entities": [
                {"start": 12, "end": 15, "label": "name"},
                {"start": 16, "end": 24, "label": "surname"},
            ],
        }
    ]

    enhanced = apply_to_dataset(sample_dataset)
    print(f"Przed: {len(sample_dataset[0]['entities'])} entities")
    print(f"Po: {len(enhanced[0]['entities'])} entities")
    for ent in enhanced[0]["entities"]:
        value = enhanced[0]["text"][ent["start"] : ent["end"]]
        print(f"  {ent['label']}: {value}")
