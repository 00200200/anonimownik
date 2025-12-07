import argparse
import json
from pathlib import Path

from src.models.anonymizer import Anonymizer


def run(text: str, model_path: str, show_spans: bool) -> None:
    anonymizer = Anonymizer(model_path=model_path)
    spans = anonymizer.predict(text)
    anonymized = anonymizer.anonymize(text)

    print("Tekst wejściowy:")
    print(text)
    if show_spans:
        print("\nWykryte encje:")
        if spans:
            for ent in spans:
                segment = text[ent["start"] : ent["end"]]
                print(
                    f"- {ent['label']} [{ent['start']}:{ent['end']}] -> \"{segment}\""
                )
        else:
            print("- (brak)")
    print("\nZanonimizowany tekst:")
    print(anonymized)
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Szybki test anonimizatora.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/test.json"),
        help="Ścieżka do pliku JSONL z danymi testowymi.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Ile pierwszych przykładów z pliku wczytać.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/herbert-ner"),
        help="Ścieżka do wytrenowanego modelu.",
    )
    parser.add_argument(
        "--show-spans",
        action="store_true",
        help="Czy wypisać wykryte encje przed anonimizacją.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Brak pliku z danymi: {args.data}")

    rows = []
    with args.data.open() as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            rows.append(json.loads(line))

    for row in rows:
        run(row["text"], str(args.model_path), args.show_spans)


if __name__ == "__main__":
    main()
