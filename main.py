"""Główny skrypt anonimizacji tekstu"""

from src.masks import mask_text, find_entities


def anonymize(text: str) -> str:
    """
    Anonimizuje tekst zamieniając wykryte dane na tokeny.

    Args:
        text: Tekst do zanonimizowania

    Returns:
        Zamaskowany tekst
    """
    masked, entities = mask_text(text)
    return masked


def anonymize_with_details(text: str) -> dict:
    """
    Anonimizuje tekst i zwraca szczegóły.

    Returns:
        {
            "original": tekst oryginalny,
            "anonymized": tekst zamaskowany,
            "entities": lista wykrytych entities
        }
    """
    masked, entities = mask_text(text)
    return {"original": text, "anonymized": masked, "entities": entities}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Podaj tekst do zanonimizowania: ")

    result = anonymize_with_details(text)

    print("\nOryginalny:")
    print(result["original"])

    print("\nZanonimizowany:")
    print(result["anonymized"])

    if result["entities"]:
        print(f"\nWykryte dane ({len(result['entities'])}):")
        for ent in result["entities"]:
            value = result["original"][ent["start"] : ent["end"]]
            print(f"  {ent['label']}: {value}")
