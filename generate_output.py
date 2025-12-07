#!/usr/bin/env python3
"""
Skrypt do generowania pliku output dla konkursu ChallengeRocket.
Zespół: SolvroGen

Użycie:
    python generate_output.py input.txt output_solvrogen.txt
    python generate_output.py --help
"""

import argparse
import re
import sys
import time
from pathlib import Path

# Dodaj ścieżkę do modułów
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import create_pipeline


def convert_tags_to_square_brackets(text: str) -> str:
    """
    Konwertuje tagi z formatu {name} na [name].
    
    Pipeline używa wewnętrznie {name}, ale konkurencja wymaga [name].
    """
    # Zamień {tag} na [tag]
    return re.sub(r'\{([^}]+)\}', r'[\1]', text)


def process_file(input_path: str, output_path: str, model_path: str = None, verbose: bool = False):
    """
    Przetwarza plik wejściowy i generuje zanonimizowany output.
    
    Args:
        input_path: Ścieżka do pliku wejściowego
        output_path: Ścieżka do pliku wyjściowego
        model_path: Opcjonalna ścieżka do modelu NER
        verbose: Czy wyświetlać szczegółowe informacje
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"Błąd: Plik wejściowy nie istnieje: {input_path}")
        sys.exit(1)
    
    # Utwórz pipeline
    print("Ładowanie modelu...")
    pipeline = create_pipeline(model_path=model_path)
    print("Model załadowany.")
    
    # Wczytaj plik wejściowy
    print(f"Wczytywanie pliku: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Liczba linii do przetworzenia: {len(lines)}")
    
    # Przetwórz każdą linię
    print("Przetwarzanie...")
    start_time = time.time()
    
    output_lines = []
    total_entities = 0
    
    for i, line in enumerate(lines):
        # Zachowaj puste linie
        if not line.strip():
            output_lines.append(line)
            continue
        
        # Anonimizuj linię
        result = pipeline.anonymize_only(line.rstrip('\n'))
        
        # Konwertuj format tagów {name} -> [name]
        anon_text = convert_tags_to_square_brackets(result.anon_text)
        
        output_lines.append(anon_text + '\n')
        total_entities += len(result.entities_meta)
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  Przetworzono {i + 1}/{len(lines)} linii...")
    
    processing_time = time.time() - start_time
    
    # Zapisz wynik
    print(f"Zapisywanie do: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    # Podsumowanie
    print("\n" + "=" * 60)
    print("PODSUMOWANIE")
    print("=" * 60)
    print(f"Przetworzonych linii: {len(lines)}")
    print(f"Wykrytych encji: {total_entities}")
    print(f"Czas przetwarzania: {processing_time:.2f} sekund")
    print(f"Średni czas na linię: {(processing_time / len(lines) * 1000):.2f} ms")
    print(f"Plik wyjściowy: {output_path}")
    print("=" * 60)
    
    return processing_time, len(lines), total_entities


def main():
    parser = argparse.ArgumentParser(
        description='Generator pliku output dla konkursu ChallengeRocket - Zespół SolvroGen',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
    python generate_output.py input.txt output_solvrogen.txt
    python generate_output.py --model models/herbert-ner input.txt output.txt
    python generate_output.py -v input.txt output.txt
        """
    )
    
    parser.add_argument('input', help='Ścieżka do pliku wejściowego')
    parser.add_argument('output', help='Ścieżka do pliku wyjściowego')
    parser.add_argument('--model', '-m', help='Ścieżka do modelu NER (opcjonalnie)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Szczegółowe informacje')
    
    args = parser.parse_args()
    
    process_file(args.input, args.output, args.model, args.verbose)


if __name__ == '__main__':
    main()

