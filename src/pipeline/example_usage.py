"""Przykład użycia generatora syntetycznego."""

from src.pipeline.synthetic_generator import SyntheticGenerator


def example_basic():
    """Podstawowy przykład użycia."""
    generator = SyntheticGenerator(seed=42)
    
    # Przykład 1: Proste użycie z anon_text
    anon_text = "Nazywam się {name} {surname} i mieszkam w {city}."
    result = generator.generate_from_anon_text(anon_text)
    print("Przykład 1:")
    print(f"  Wejście: {anon_text}")
    print(f"  Wynik:   {result}")
    print()
    
    # Przykład 2: Użycie z entities_meta (pełna kontrola)
    anon_text2 = "Nazywam się {name} {surname}."
    entities_meta = [
        {
            "span": (13, 19),  # pozycja tokena {name}
            "entity_type": "name",
            "original_text": "Anna",
            "morph_features": {"case": "nom", "number": "sg", "gender": "f"}
        },
        {
            "span": (20, 28),  # pozycja tokena {surname}
            "entity_type": "surname",
            "original_text": "Kowalska",
            "morph_features": {"case": "nom", "number": "sg", "gender": "f"}
        }
    ]
    
    result2 = generator.generate(anon_text2, entities_meta)
    print("Przykład 2:")
    print(f"  Wejście: {anon_text2}")
    print(f"  Wynik:   {result2}")
    print()
    
    # Przykład 3: Z odmianą (dopełniacz)
    anon_text3 = "To jest dom {name}."
    entities_meta3 = [
        {
            "span": (13, 19),
            "entity_type": "name",
            "original_text": "Anny",  # dopełniacz
            "morph_features": {"case": "gen", "number": "sg", "gender": "f"}
        }
    ]
    
    result3 = generator.generate(anon_text3, entities_meta3)
    print("Przykład 3 (z odmianą):")
    print(f"  Wejście: {anon_text3}")
    print(f"  Wynik:   {result3}")
    print()


def example_with_custom_dictionary():
    """Przykład z własnym słownikiem."""
    generator = SyntheticGenerator(seed=42)
    
    # Dodaj własne wartości
    generator.add_entity_dictionary("name", ["Zygmunt", "Bolesław", "Władysław"])
    generator.add_entity_dictionary("city", ["Gniezno", "Płock", "Kalisz"])
    
    anon_text = "Nazywam się {name} i pochodzę z {city}."
    result = generator.generate_from_anon_text(anon_text)
    
    print("Przykład z własnym słownikiem:")
    print(f"  Wejście: {anon_text}")
    print(f"  Wynik:   {result}")
    print()


if __name__ == "__main__":
    example_basic()
    example_with_custom_dictionary()

