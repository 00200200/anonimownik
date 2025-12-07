"""Interfejs linii poleceń dla pipeline'u anonimizacji.

Użycie:
    # Interaktywny tryb
    python -m pipeline.cli
    
    # Przetwarzanie tekstu
    python -m pipeline.cli --text "Jestem Jan Kowalski z Warszawy."
    
    # Przetwarzanie pliku
    python -m pipeline.cli --input plik.txt --output wynik.json
    
    # Przetwarzanie katalogu
    python -m pipeline.cli --input-dir dane/ --output-dir wyniki/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .pipeline import AnonymizationPipeline, create_pipeline


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty linii poleceń."""
    parser = argparse.ArgumentParser(
        description="Pipeline anonimizacji i generacji syntetycznej dla języka polskiego",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  # Tryb interaktywny
  python -m pipeline.cli
  
  # Przetwarzanie tekstu z linii poleceń
  python -m pipeline.cli --text "Mam na imię Anna Kowalska."
  
  # Przetwarzanie pliku
  python -m pipeline.cli --input dane.txt --output wynik.json
  
  # Przetwarzanie katalogu
  python -m pipeline.cli --input-dir ./dokumenty --output-dir ./wyniki
  
  # Tylko anonimizacja (bez generacji syntetycznej)
  python -m pipeline.cli --text "Jan Kowalski" --anonymize-only
  
  # Z własnym modelem NER
  python -m pipeline.cli --model models/herbert-ner --text "Test"
"""
    )
    
    # Tryb wejścia
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--text', '-t',
        type=str,
        help='Tekst do przetworzenia'
    )
    input_group.add_argument(
        '--input', '-i',
        type=str,
        help='Ścieżka do pliku wejściowego'
    )
    input_group.add_argument(
        '--input-dir',
        type=str,
        help='Katalog z plikami do przetworzenia'
    )
    
    # Wyjście
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Ścieżka do pliku wyjściowego'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Katalog na pliki wyjściowe'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'text', 'both'],
        default='text',
        help='Format wyjścia (domyślnie: text)'
    )
    
    # Opcje pipeline'u
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Ścieżka do modelu NER'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        help='Ziarno losowości dla generatora syntetycznego'
    )
    parser.add_argument(
        '--no-regex',
        action='store_true',
        help='Wyłącz reguły regex do wykrywania encji'
    )
    parser.add_argument(
        '--anonymize-only',
        action='store_true',
        help='Wykonaj tylko anonimizację (bez generacji syntetycznej)'
    )
    
    # Dodatkowe
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Wyświetl szczegółowe informacje'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Wyświetl wynik jako JSON'
    )
    
    return parser.parse_args()


def run_interactive(pipeline: AnonymizationPipeline, args: argparse.Namespace) -> None:
    """Tryb interaktywny."""
    print("=" * 60)
    print("Pipeline Anonimizacji - Tryb Interaktywny")
    print("=" * 60)
    print("Wpisz tekst do przetworzenia (lub 'q' aby wyjść)")
    print()
    
    while True:
        try:
            text = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nDo widzenia!")
            break
        
        if not text:
            continue
        if text.lower() in ('q', 'quit', 'exit'):
            print("Do widzenia!")
            break
        
        result = pipeline.process(text)
        
        print()
        if args.anonymize_only:
            print(f"Anonimizowany: {result.anonymized_text}")
        else:
            print(f"Anonimizowany: {result.anonymized_text}")
            print(f"Syntetyczny:   {result.synthetic_text}")
        
        if args.verbose and result.entities:
            print(f"\nWykryte encje ({len(result.entities)}):")
            for e in result.entities:
                morph = f" [{e.morph_features}]" if e.morph_features else ""
                print(f"  - {e.entity_type}: '{e.original_text}'{morph}")
        
        print()


def process_text(pipeline: AnonymizationPipeline, text: str, args: argparse.Namespace) -> None:
    """Przetwarza pojedynczy tekst."""
    result = pipeline.process(text)
    
    if args.json or args.format == 'json':
        print(result.to_json())
    else:
        print("=== ORYGINALNY ===")
        print(result.original_text)
        print()
        print("=== ANONIMIZOWANY ===")
        print(result.anonymized_text)
        
        if not args.anonymize_only:
            print()
            print("=== SYNTETYCZNY ===")
            print(result.synthetic_text)
        
        if args.verbose and result.entities:
            print()
            print(f"=== ENCJE ({len(result.entities)}) ===")
            for e in result.entities:
                morph = f" {e.morph_features}" if e.morph_features else ""
                print(f"  [{e.start}:{e.end}] {e.entity_type}: '{e.original_text}'{morph}")


def process_file(pipeline: AnonymizationPipeline, args: argparse.Namespace) -> None:
    """Przetwarza plik."""
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Błąd: Plik '{input_path}' nie istnieje", file=sys.stderr)
        sys.exit(1)
    
    output_path = args.output
    if not output_path:
        # Domyślna nazwa wyjściowa
        suffix = '.json' if args.format == 'json' else '.anon.txt'
        output_path = input_path.with_suffix(suffix)
    
    if args.verbose:
        print(f"Przetwarzanie: {input_path}")
        print(f"Wyjście: {output_path}")
    
    result = pipeline.process_file(
        input_path,
        output_path,
        output_format=args.format,
    )
    
    print(f"Przetworzono {len(result.results)} fragmentów")
    print(f"Wykryto {result.total_entities} encji")
    
    if args.verbose:
        print("\nTypy encji:")
        for entity_type, count in sorted(result.entity_type_counts.items()):
            print(f"  {entity_type}: {count}")


def process_directory(pipeline: AnonymizationPipeline, args: argparse.Namespace) -> None:
    """Przetwarza katalog."""
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Błąd: Katalog '{input_dir}' nie istnieje", file=sys.stderr)
        sys.exit(1)
    
    output_dir = args.output_dir or str(input_dir) + "_processed"
    
    if args.verbose:
        print(f"Przetwarzanie katalogu: {input_dir}")
        print(f"Wyjście do: {output_dir}")
    
    results = pipeline.process_directory(
        input_dir,
        output_dir,
        output_format=args.format,
    )
    
    total_entities = sum(r.total_entities for r in results.values())
    total_fragments = sum(len(r.results) for r in results.values())
    
    print(f"Przetworzono {len(results)} plików")
    print(f"Łącznie {total_fragments} fragmentów")
    print(f"Wykryto {total_entities} encji")


def main() -> None:
    """Główna funkcja CLI."""
    args = parse_args()
    
    # Utwórz pipeline
    pipeline = create_pipeline(
        model_path=args.model,
        seed=args.seed,
        use_regex_rules=not args.no_regex,
    )
    
    # Wybierz tryb działania
    if args.text:
        process_text(pipeline, args.text, args)
    elif args.input:
        process_file(pipeline, args)
    elif args.input_dir:
        process_directory(pipeline, args)
    else:
        # Tryb interaktywny
        run_interactive(pipeline, args)


if __name__ == "__main__":
    main()

