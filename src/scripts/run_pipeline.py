import argparse
import sys
import time
from pathlib import Path

from src.pipeline import create_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline anonimizacji dla języka polskiego"
    )
    parser.add_argument("text", nargs="?", help="Tekst do przetworzenia")
    parser.add_argument("--file", "-f", help="Plik wejściowy")
    parser.add_argument("--output", "-o", help="Plik wyjściowy")
    parser.add_argument(
        "--model", "-m", default="models/herbert-ner", help="Ścieżka do modelu NER"
    )
    parser.add_argument("--seed", "-s", type=int, help="Ziarno losowości")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Więcej informacji"
    )

    args = parser.parse_args()

    # Sprawdź czy model istnieje
    model_path = args.model if Path(args.model).exists() else None

    print("Inicjalizacja pipeline'u...")
    pipeline = create_pipeline(
        model_path=model_path,
        seed=args.seed,
    )
    print("Pipeline gotowy!\n")

    if args.file:
        # Przetwarzanie pliku
        input_path = Path(args.file)
        if not input_path.exists():
            print(f"Błąd: Plik '{input_path}' nie istnieje")
            sys.exit(1)

        output_path = args.output or str(input_path.with_suffix(".processed.txt"))
        t0 = time.perf_counter()
        result = pipeline.process_file(input_path, output_path, output_format="both")
        total_s = time.perf_counter() - t0
        per_fragment = total_s / len(result.results) if result.results else 0.0

        print(f"Przetworzono {len(result.results)} fragmentów")
        print(f"Wykryto {result.total_entities} encji")
        print(f"Czas: {total_s:.3f}s (średnio {per_fragment*1000:.2f} ms/fragment)")
        print(f"Wynik zapisany do: {output_path}")

    elif args.text:
        # Pojedynczy tekst
        t0 = time.perf_counter()
        result = pipeline.process(args.text)
        total_s = time.perf_counter() - t0
        per_fragment = total_s  # pojedynczy tekst

        print("ORYGINALNY:")
        print(f"  {result.original_text}")
        print()
        print("ANONIMIZOWANY:")
        print(f"  {result.anonymized_text}")
        print()
        print("SYNTETYCZNY:")
        print(f"  {result.synthetic_text}")
        print()
        print(f"Czas: {total_s:.3f}s (na tekst)")

        if args.verbose and result.entities:
            print()
            print("ENCJE:")
            for e in result.entities:
                print(f"  - {e.entity_type}: '{e.original_text}' {e.morph_features}")

    else:
        # Tryb interaktywny
        print("=" * 50)
        print("Tryb interaktywny - wpisz tekst lub 'q' aby wyjść")
        print("=" * 50)

        while True:
            try:
                text = input("\n>>> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nDo widzenia!")
                break

            if not text:
                continue
            if text.lower() in ("q", "quit", "exit"):
                print("Do widzenia!")
                break

            result = pipeline.process(text)

            print(f"\nAnonimizowany: {result.anonymized_text}")
            print(f"Syntetyczny:   {result.synthetic_text}")

            if args.verbose:
                for e in result.entities:
                    print(f"  [{e.entity_type}] '{e.original_text}'")


if __name__ == "__main__":
    main()
