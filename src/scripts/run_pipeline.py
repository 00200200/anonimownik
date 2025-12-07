#!/usr/bin/env python3
"""Skrypt do uruchamiania pipeline'u anonimizacji.

Użycie:
    # Tryb interaktywny
    python -m scripts.run_pipeline
    
    # Z tekstem
    python -m scripts.run_pipeline "Jan Kowalski mieszka w Warszawie."
    
    # Przetwarzanie pliku
    python -m scripts.run_pipeline --file dane.txt --output wynik.json
"""

import argparse
import sys
from pathlib import Path

# Dodaj ścieżkę do src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import AnonymizationPipeline, create_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline anonimizacji dla języka polskiego"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Tekst do przetworzenia"
    )
    parser.add_argument(
        "--file", "-f",
        help="Plik wejściowy"
    )
    parser.add_argument(
        "--output", "-o",
        help="Plik wyjściowy"
    )
    parser.add_argument(
        "--model", "-m",
        default="models/herbert-ner",
        help="Ścieżka do modelu NER"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        help="Ziarno losowości"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Więcej informacji"
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
        
        output_path = args.output or str(input_path.with_suffix('.processed.txt'))
        
        result = pipeline.process_file(input_path, output_path, output_format='both')
        
        print(f"Przetworzono {len(result.results)} fragmentów")
        print(f"Wykryto {result.total_entities} encji")
        print(f"Wynik zapisany do: {output_path}")
        
    elif args.text:
        # Pojedynczy tekst
        result = pipeline.process(args.text)
        
        print("ORYGINALNY:")
        print(f"  {result.original_text}")
        print()
        print("ANONIMIZOWANY:")
        print(f"  {result.anonymized_text}")
        print()
        print("SYNTETYCZNY:")
        print(f"  {result.synthetic_text}")
        
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
            if text.lower() in ('q', 'quit', 'exit'):
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

