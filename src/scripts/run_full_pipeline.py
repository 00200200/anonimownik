import argparse
import sys
import time
from pathlib import Path

from src.pipeline import create_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Uruchamia pełny pipeline (anonimizacja + generacja) i zapisuje wyłącznie wynik końcowy."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Plik wejściowy (np. orig_final.txt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("anonim_final.txt"),
        help="Plik wyjściowy z wynikiem końcowym (synthetic).",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        default=Path("models/herbert-ner"),
        help="Ścieżka do modelu NER.",
    )
    parser.add_argument(
        "--seed", "-s", type=int, help="Ziarno generatora (powtarzalność)."
    )
    parser.add_argument("--no-regex", action="store_true", help="Wyłącz reguły regex.")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Brak pliku wejściowego: {args.input}", file=sys.stderr)
        sys.exit(1)

    model_path = str(args.model) if args.model.exists() else None
    pipeline = create_pipeline(
        model_path=model_path, seed=args.seed, use_regex_rules=not args.no_regex
    )

    lines = args.input.read_text(encoding="utf-8").splitlines()

    out_lines = []
    total_entities = 0
    t0 = time.perf_counter()

    for line in lines:
        if not line.strip():
            out_lines.append("")
            continue
        res = pipeline.process(line)
        total_entities += len(res.entities)
        out_lines.append(res.synthetic_text)

    elapsed = time.perf_counter() - t0
    args.output.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"Zapisano: {args.output} (linie: {len(out_lines)}, encje: {total_entities})")
    print(
        f"Czas całkowity: {elapsed:.3f}s (średnio {elapsed/len(out_lines):.4f}s/linia)"
    )


if __name__ == "__main__":
    main()
