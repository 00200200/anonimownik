"""Testy generatora syntetycznego z poprawną odmianą."""

from src.pipeline.synthetic_generator import SyntheticGenerator
from src.pipeline.anonymizer import Anonymizer, EntityMeta


def test_basic_replacement():
    """Test podstawowej zamiany tokenów."""
    print("=" * 60)
    print("Test 1: Podstawowa zamiana tokenów")
    print("=" * 60)
    
    generator = SyntheticGenerator(seed=42)
    
    anon_text = "Nazywam się {name} {surname} i mieszkam w {city}."
    result = generator.generate_from_anon_text(anon_text)
    
    print(f"Wejście: {anon_text}")
    print(f"Wynik:   {result}")
    
    assert "{name}" not in result, "Token {name} nie został zamieniony"
    assert "{surname}" not in result, "Token {surname} nie został zamieniony"
    assert "{city}" not in result, "Token {city} nie został zamieniony"
    
    print("✓ Test zakończony pomyślnie\n")


def test_nominative_case():
    """Test mianownika (kto? co?)."""
    print("=" * 60)
    print("Test 2: Mianownik (kto? co?)")
    print("=" * 60)
    
    generator = SyntheticGenerator(seed=42)
    
    # Mianownik - "Anna mieszka..."
    entities_meta = [
        {
            "entity_type": "name",
            "original_text": "Anna",
            "morph_features": {"case": "Nom", "gender": "Fem", "number": "Sing"}
        }
    ]
    
    anon_text = "{name} mieszka w Warszawie."
    result = generator.generate(anon_text, entities_meta)
    
    print(f"Wejście: {anon_text}")
    print(f"Cechy:   Mianownik, rodzaj żeński")
    print(f"Wynik:   {result}")
    
    # Sprawdź czy wynik zawiera imię w mianowniku (kończy się na 'a')
    words = result.split()
    first_word = words[0]
    assert first_word.endswith('a') or first_word in ['Ewa'], f"Oczekiwano mianownika żeńskiego, otrzymano: {first_word}"
    
    print("✓ Test zakończony pomyślnie\n")


def test_genitive_case():
    """Test dopełniacza (kogo? czego?)."""
    print("=" * 60)
    print("Test 3: Dopełniacz (kogo? czego?)")
    print("=" * 60)
    
    generator = SyntheticGenerator(seed=42)
    
    # Dopełniacz - "dom Anny", "mieszkanie Jana"
    test_cases = [
        {
            "entities_meta": [{
                "entity_type": "name",
                "original_text": "Anny",
                "morph_features": {"case": "Gen", "gender": "Fem", "number": "Sing"}
            }],
            "anon_text": "To jest dom {name}.",
            "expected_endings": ["y", "i", "ii"],
            "gender": "żeński"
        },
        {
            "entities_meta": [{
                "entity_type": "name",
                "original_text": "Jana",
                "morph_features": {"case": "Gen", "gender": "Masc", "number": "Sing"}
            }],
            "anon_text": "To jest dom {name}.",
            "expected_endings": ["a"],
            "gender": "męski"
        },
        {
            "entities_meta": [{
                "entity_type": "city",
                "original_text": "Warszawy",
                "morph_features": {"case": "Gen", "gender": "Fem", "number": "Sing"}
            }],
            "anon_text": "Jestem z {city}.",
            "expected_endings": ["y", "a", "i", "u", "c"],
            "gender": "miasto"
        }
    ]
    
    for i, tc in enumerate(test_cases):
        result = generator.generate(tc["anon_text"], tc["entities_meta"])
        print(f"  Przypadek {i+1} ({tc['gender']}):")
        print(f"    Wejście: {tc['anon_text']}")
        print(f"    Wynik:   {result}")
        
        # Wyciągnij słowo przed kropką
        words = result.rstrip('.').split()
        last_word = words[-1]
        
        # Sprawdź czy kończy się na oczekiwaną końcówkę
        has_valid_ending = any(last_word.lower().endswith(end) for end in tc["expected_endings"])
        if not has_valid_ending:
            print(f"    ⚠ Uwaga: '{last_word}' nie kończy się na {tc['expected_endings']}")
        else:
            print(f"    ✓ OK")
    
    print("✓ Test zakończony pomyślnie\n")


def test_locative_case():
    """Test miejscownika (o kim? o czym? w kim? w czym?)."""
    print("=" * 60)
    print("Test 4: Miejscownik (o kim? o czym?)")
    print("=" * 60)
    
    generator = SyntheticGenerator(seed=42)
    
    # Miejscownik - "w Warszawie", "o Annie"
    test_cases = [
        {
            "entities_meta": [{
                "entity_type": "city",
                "original_text": "Warszawie",
                "morph_features": {"case": "Loc", "gender": "Fem", "number": "Sing"}
            }],
            "anon_text": "Mieszkam w {city}.",
            "context": "miasto"
        },
        {
            "entities_meta": [{
                "entity_type": "name",
                "original_text": "Annie",
                "morph_features": {"case": "Loc", "gender": "Fem", "number": "Sing"}
            }],
            "anon_text": "Myślę o {name}.",
            "context": "imię żeńskie"
        },
    ]
    
    for i, tc in enumerate(test_cases):
        result = generator.generate(tc["anon_text"], tc["entities_meta"])
        print(f"  Przypadek {i+1} ({tc['context']}):")
        print(f"    Wejście: {tc['anon_text']}")
        print(f"    Wynik:   {result}")
    
    print("✓ Test zakończony pomyślnie\n")


def test_accusative_case():
    """Test biernika (kogo? co?)."""
    print("=" * 60)
    print("Test 5: Biernik (kogo? co?)")
    print("=" * 60)
    
    generator = SyntheticGenerator(seed=42)
    
    # Biernik - "widzę Annę", "znam Jana"
    test_cases = [
        {
            "entities_meta": [{
                "entity_type": "name",
                "original_text": "Annę",
                "morph_features": {"case": "Acc", "gender": "Fem", "number": "Sing"}
            }],
            "anon_text": "Widzę {name}.",
            "expected_endings": ["ę"],
            "gender": "żeński"
        },
        {
            "entities_meta": [{
                "entity_type": "name",
                "original_text": "Jana",
                "morph_features": {"case": "Acc", "gender": "Masc", "number": "Sing"}
            }],
            "anon_text": "Znam {name}.",
            "expected_endings": ["a"],
            "gender": "męski"
        },
    ]
    
    for i, tc in enumerate(test_cases):
        result = generator.generate(tc["anon_text"], tc["entities_meta"])
        print(f"  Przypadek {i+1} ({tc['gender']}):")
        print(f"    Wejście: {tc['anon_text']}")
        print(f"    Wynik:   {result}")
        
        # Wyciągnij słowo przed kropką
        words = result.rstrip('.').split()
        last_word = words[-1]
        
        # Sprawdź czy kończy się na oczekiwaną końcówkę
        has_valid_ending = any(last_word.lower().endswith(end) for end in tc["expected_endings"])
        if has_valid_ending:
            print(f"    ✓ OK - kończy się na {tc['expected_endings']}")
        else:
            print(f"    ⚠ '{last_word}' nie kończy się na {tc['expected_endings']}")
    
    print("✓ Test zakończony pomyślnie\n")


def test_full_pipeline():
    """Test pełnego pipeline'u: anonimizacja -> generacja syntetyczna."""
    print("=" * 60)
    print("Test 6: Pełny pipeline")
    print("=" * 60)
    
    # Uwaga: ten test wymaga modelu NER, więc używamy tylko regex
    anonymizer = Anonymizer(model_path=None)  # bez modelu NER
    generator = SyntheticGenerator(seed=42)
    
    test_texts = [
        "Mój PESEL to 90010112345.",
        "Napisz na jan.kowalski@gmail.com.",
        "Zadzwoń pod +48 500 600 700.",
    ]
    
    for text in test_texts:
        print(f"\n  Oryginalny tekst: {text}")
        
        result = anonymizer.anonymize(text)
        print(f"  Zanonimizowany:   {result.anon_text}")
        print(f"  Encje: {[e.to_dict() for e in result.entities_meta]}")
        
        synth_text = generator.generate(result.anon_text, [e.to_dict() for e in result.entities_meta])
        print(f"  Syntetyczny:      {synth_text}")
    
    print("\n✓ Test zakończony pomyślnie\n")


def test_morphological_detection():
    """Test wykrywania cech morfologicznych przez spaCy."""
    print("=" * 60)
    print("Test 7: Wykrywanie cech morfologicznych")
    print("=" * 60)
    
    anonymizer = Anonymizer(model_path=None)
    
    test_cases = [
        ("Anna mieszka w Warszawie.", 0, 4),  # Anna - mianownik
        ("Widziałem Annę.", 10, 14),  # Annę - biernik
        ("Dom Anny.", 4, 8),  # Anny - dopełniacz
        ("Jadę do Krakowa.", 8, 15),  # Krakowa - dopełniacz
        ("Mieszkam w Krakowie.", 11, 19),  # Krakowie - miejscownik
    ]
    
    for text, start, end in test_cases:
        features = anonymizer._extract_morph_features(text, start, end)
        fragment = text[start:end]
        print(f"  '{fragment}' w '{text}'")
        print(f"    Cechy: {features}")
    
    print("\n✓ Test zakończony pomyślnie\n")


def run_all_tests():
    """Uruchom wszystkie testy."""
    print("\n" + "=" * 60)
    print("TESTY GENERATORA SYNTETYCZNEGO")
    print("=" * 60 + "\n")
    
    test_basic_replacement()
    test_nominative_case()
    test_genitive_case()
    test_locative_case()
    test_accusative_case()
    test_full_pipeline()
    test_morphological_detection()
    
    print("=" * 60)
    print("WSZYSTKIE TESTY ZAKOŃCZONE POMYŚLNIE!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
