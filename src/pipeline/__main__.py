"""Entry point dla uruchamiania pipeline'u jako modułu.

Użycie:
    python -m pipeline --help
    python -m pipeline --text "Jan Kowalski mieszka w Warszawie."
    python -m pipeline --input plik.txt --output wynik.json
"""

from .cli import main

if __name__ == "__main__":
    main()

