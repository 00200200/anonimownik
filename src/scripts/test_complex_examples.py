"""Trudniejsze przykłady testowe dla pipeline'u anonimizacji i generacji syntetycznej."""

from src.pipeline import Anonymizer, SyntheticGenerator


def test_complex_sentences():
    """Testy ze złożonymi zdaniami i wieloma encjami."""
    print("=" * 70)
    print("TRUDNIEJSZE PRZYKŁADY - ZŁOŻONE ZDANIA")
    print("=" * 70)
    
    anonymizer = Anonymizer(model_path=None)
    generator = SyntheticGenerator(seed=42)
    
    complex_texts = [
        # Wiele encji w jednym zdaniu
        "Napisz do mnie na jan.kowalski@firma.pl lub zadzwoń pod 501-234-567.",
        
        # Dane wrażliwe z kontekstem
        "Mój PESEL to 90010112345, a numer konta to 12 3456 7890 1234 5678 9012 3456.",
        
        # Email z różnymi formatami
        "Kontakt: test.user123@subdomain.example.com, backup: info@company.co.uk",
        
        # Telefony w różnych formatach
        "Biuro: +48 22 123 45 67, komórka: 600-700-800, fax: 123 456 789.",
        
        # Mieszanka danych
        "Przelew na konto 45 1140 2004 0000 3202 1234 5678, kontakt: biuro@example.pl, tel: +48 500 600 700.",
    ]
    
    for i, text in enumerate(complex_texts, 1):
        print(f"\n{'─' * 70}")
        print(f"Przykład {i}:")
        print(f"  Oryginalny:     {text}")
        
        result = anonymizer.anonymize(text)
        print(f"  Zanonimizowany: {result.anon_text}")
        print(f"  Wykryte encje:  {len(result.entities_meta)}")
        for e in result.entities_meta:
            print(f"    - {e.entity_type}: '{e.original_text}'")
        
        synth = generator.generate(result.anon_text, [e.to_dict() for e in result.entities_meta])
        print(f"  Syntetyczny:    {synth}")


def test_morphological_cases():
    """Testy różnych przypadków gramatycznych w kontekście."""
    print("\n" + "=" * 70)
    print("TRUDNIEJSZE PRZYKŁADY - ODMIANA MORFOLOGICZNA")
    print("=" * 70)
    
    generator = SyntheticGenerator(seed=123)
    
    # Każdy przypadek w kontekście zdania
    cases_examples = [
        # Mianownik (Nominativus) - kto? co?
        {
            "text": "{name} {surname} jest najlepszym pracownikiem w {company}.",
            "entities": [
                {"entity_type": "name", "original_text": "Anna", "morph_features": {"case": "Nom", "gender": "Fem"}},
                {"entity_type": "surname", "original_text": "Kowalska", "morph_features": {"case": "Nom", "gender": "Fem"}},
                {"entity_type": "company", "original_text": "Allegro", "morph_features": {}},
            ],
            "case": "Mianownik"
        },
        # Dopełniacz (Genitivus) - kogo? czego?
        {
            "text": "Szukam {name} {surname} z {city}.",
            "entities": [
                {"entity_type": "name", "original_text": "Anny", "morph_features": {"case": "Gen", "gender": "Fem"}},
                {"entity_type": "surname", "original_text": "Kowalskiej", "morph_features": {"case": "Gen", "gender": "Fem"}},
                {"entity_type": "city", "original_text": "Warszawy", "morph_features": {"case": "Gen"}},
            ],
            "case": "Dopełniacz"
        },
        # Celownik (Dativus) - komu? czemu?
        {
            "text": "Przekażę dokumenty {name} {surname}.",
            "entities": [
                {"entity_type": "name", "original_text": "Janowi", "morph_features": {"case": "Dat", "gender": "Masc"}},
                {"entity_type": "surname", "original_text": "Kowalskiemu", "morph_features": {"case": "Dat", "gender": "Masc"}},
            ],
            "case": "Celownik"
        },
        # Biernik (Accusativus) - kogo? co?
        {
            "text": "Widziałem {name} {surname} jadącą do {city}.",
            "entities": [
                {"entity_type": "name", "original_text": "Annę", "morph_features": {"case": "Acc", "gender": "Fem"}},
                {"entity_type": "surname", "original_text": "Kowalską", "morph_features": {"case": "Acc", "gender": "Fem"}},
                {"entity_type": "city", "original_text": "Kraków", "morph_features": {"case": "Acc"}},
            ],
            "case": "Biernik"
        },
        # Narzędnik (Instrumentalis) - z kim? z czym?
        {
            "text": "Rozmawiałem z {name} {surname} o nowym projekcie.",
            "entities": [
                {"entity_type": "name", "original_text": "Piotrem", "morph_features": {"case": "Ins", "gender": "Masc"}},
                {"entity_type": "surname", "original_text": "Nowakiem", "morph_features": {"case": "Ins", "gender": "Masc"}},
            ],
            "case": "Narzędnik"
        },
        # Miejscownik (Locativus) - o kim? o czym? w kim? w czym?
        {
            "text": "Mieszkam w {city} i myślę o {name}.",
            "entities": [
                {"entity_type": "city", "original_text": "Krakowie", "morph_features": {"case": "Loc"}},
                {"entity_type": "name", "original_text": "Annie", "morph_features": {"case": "Loc", "gender": "Fem"}},
            ],
            "case": "Miejscownik"
        },
    ]
    
    for example in cases_examples:
        print(f"\n{'─' * 70}")
        print(f"Przypadek: {example['case']}")
        print(f"  Wzorzec:    {example['text']}")
        
        result = generator.generate(example['text'], example['entities'])
        print(f"  Wynik:      {result}")


def test_edge_cases():
    """Testy brzegowe i przypadki specjalne."""
    print("\n" + "=" * 70)
    print("TRUDNIEJSZE PRZYKŁADY - PRZYPADKI BRZEGOWE")
    print("=" * 70)
    
    anonymizer = Anonymizer(model_path=None)
    generator = SyntheticGenerator(seed=42)
    
    edge_cases = [
        # Pusty tekst
        ("Pusty tekst", ""),
        
        # Tekst bez encji
        ("Bez encji", "To jest zwykłe zdanie bez żadnych danych osobowych."),
        
        # Tylko cyfry (potencjalnie PESEL lub telefon)
        ("Krótki numer", "Kod: 12345"),
        
        # Email na końcu zdania
        ("Email z kropką", "Napisz na test@example.com."),
        
        # Wiele emaili obok siebie
        ("Wiele emaili", "CC: a@b.pl, c@d.pl, e@f.pl"),
        
        # Telefon w nawiasie
        ("Telefon w nawiasie", "Zadzwoń (500 600 700) w godz. 9-17."),
        
        # Mieszane formaty
        ("Mieszane", "Tel: +48500600700, email:test@x.pl, PESEL:90010112345"),
        
        # Unicode i polskie znaki
        ("Polski tekst", "Zadzwoń do Żółwia: 123-456-789 lub napisz na żółw@łąka.pl"),
        
        # Bardzo długi tekst
        ("Długi tekst", "To jest bardzo długie zdanie " * 10 + "z emailem test@example.com na końcu."),
    ]
    
    for name, text in edge_cases:
        print(f"\n{'─' * 70}")
        print(f"Test: {name}")
        print(f"  Wejście: {text[:80]}{'...' if len(text) > 80 else ''}")
        
        if text:
            result = anonymizer.anonymize(text)
            print(f"  Zanonimizowany: {result.anon_text[:80]}{'...' if len(result.anon_text) > 80 else ''}")
            print(f"  Liczba encji: {len(result.entities_meta)}")
            
            if result.entities_meta:
                synth = generator.generate(result.anon_text, [e.to_dict() for e in result.entities_meta])
                print(f"  Syntetyczny: {synth[:80]}{'...' if len(synth) > 80 else ''}")
        else:
            print("  (pusty tekst - pominięto)")


def test_realistic_documents():
    """Testy z realistycznymi dokumentami."""
    print("\n" + "=" * 70)
    print("TRUDNIEJSZE PRZYKŁADY - REALISTYCZNE DOKUMENTY")
    print("=" * 70)
    
    anonymizer = Anonymizer(model_path=None)
    generator = SyntheticGenerator(seed=42)
    
    documents = [
        # Email biznesowy
        """Szanowny Panie,
W nawiązaniu do rozmowy telefonicznej z dnia 15.03.2024, przesyłam dane do przelewu:
Numer konta: 12 3456 7890 1234 5678 9012 3456
W razie pytań proszę o kontakt: jan.kowalski@firma.pl lub 501-234-567.
Z poważaniem""",
        
        # Formularz kontaktowy
        """DANE KONTAKTOWE:
Email: anna.nowak@gmail.com
Telefon komórkowy: +48 600 700 800
Telefon stacjonarny: 22 123 45 67
PESEL: 85032112345""",
        
        # Reklamacja
        """Zgłaszam reklamację produktu zakupionego 10.01.2024.
Numer zamówienia: ABC123456
Proszę o kontakt zwrotny: klient@poczta.pl, tel: 500-100-200.
Numer konta do zwrotu: 98 7654 3210 9876 5432 1098 7654""",
    ]
    
    for i, doc in enumerate(documents, 1):
        print(f"\n{'─' * 70}")
        print(f"Dokument {i}:")
        print(f"  Oryginalny:")
        for line in doc.strip().split('\n'):
            print(f"    {line}")
        
        result = anonymizer.anonymize(doc)
        print(f"\n  Zanonimizowany:")
        for line in result.anon_text.strip().split('\n'):
            print(f"    {line}")
        
        print(f"\n  Wykryte encje ({len(result.entities_meta)}):")
        for e in result.entities_meta:
            print(f"    - {e.entity_type}: '{e.original_text}'")
        
        synth = generator.generate(result.anon_text, [e.to_dict() for e in result.entities_meta])
        print(f"\n  Syntetyczny:")
        for line in synth.strip().split('\n'):
            print(f"    {line}")


def run_all():
    """Uruchom wszystkie trudniejsze testy."""
    test_complex_sentences()
    test_morphological_cases()
    test_edge_cases()
    test_realistic_documents()
    
    print("\n" + "=" * 70)
    print("WSZYSTKIE TRUDNIEJSZE PRZYKŁADY ZAKOŃCZONE!")
    print("=" * 70)


if __name__ == "__main__":
    run_all()

