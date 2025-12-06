#!/usr/bin/env python3
"""
Skrypt do deanonimizacji tekstu przy użyciu modelu Herbert (fill-mask).
Maksymalne wykorzystanie modelu Herbert do generowania wszystkich typów danych:
- Imiona, nazwiska, miasta, firmy - pełne przewidywanie przez Herbert
- Nazwy ulic - generowane przez Herbert z odpowiednim kontekstem
- Płeć - przewidywana przez Herbert na podstawie kontekstu
- Email - używa imion/nazwisk wygenerowanych przez Herbert
- Dane numeryczne (PESEL, telefony, daty) - algorytmiczne z kontekstem Herbert
"""

import re
import random
import argparse
from datetime import datetime, timedelta
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


class HerbertDeanonymizer:
    def __init__(self, model_name: str = "allegro/herbert-base-cased"):
        """Inicjalizacja deanonimizatora z modelem Herbert."""
        print(f"Ładowanie modelu {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        
        self.street_types = ["ul.", "al.", "plac", "os."]
        
    def predict_with_context(self, full_text: str, mask_position: int, window: int = 100, top_k: int = 5, filter_func=None, fallback: str = "Nieznany") -> str:
        """Użyj Herberta z pełnym kontekstem otaczającego tekstu."""
        try:
            # Wyciągnij fragment tekstu wokół maski
            start = max(0, mask_position - window)
            end = min(len(full_text), mask_position + window)
            context_text = full_text[start:end]
            
            # Znajdź pozycję [MASK] w kontekście
            if "[MASK]" not in context_text:
                return fallback
            
            masked_text = context_text.replace("[MASK]", self.tokenizer.mask_token)
            inputs = self.tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits
            
            mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            
            if len(mask_token_index) == 0:
                return fallback
            
            mask_token_index = mask_token_index[0]
            mask_token_logits = predictions[0, mask_token_index, :]
            
            # Pobierz więcej kandydatów do filtrowania
            top_tokens = torch.topk(mask_token_logits, min(top_k * 5, 200), dim=0).indices.tolist()
            
            candidates = []
            all_words = []
            
            for token in top_tokens:
                word = self.tokenizer.decode([token]).strip()
                word = word.replace("##", "").strip()
                
                if word and len(word) > 1 and word.replace("-", "").replace("'", "").isalpha():
                    all_words.append(word)
                    if filter_func is None or filter_func(word):
                        candidates.append(word)
                        if len(candidates) >= top_k:
                            break
            
            word_list = candidates if candidates else (all_words[:top_k] if all_words else [fallback])
            word = random.choice(word_list)
            
            return word[0].upper() + word[1:].lower() if len(word) > 1 else word.upper()
            
        except Exception as e:
            print(f"Błąd w predykcji Herbert: {e}")
            return fallback
    
    def predict_with_herbert(self, text: str, top_k: int = 5, filter_func=None, fallback: str = "Nieznany") -> str:
        """Użyj modelu Herbert do przewidzenia maskowanego słowa."""
        try:
            masked_text = text.replace("[MASK]", self.tokenizer.mask_token)
            inputs = self.tokenizer(masked_text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits
            
            mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            
            if len(mask_token_index) == 0:
                return fallback
            
            mask_token_index = mask_token_index[0]
            mask_token_logits = predictions[0, mask_token_index, :]
            
            # Pobierz więcej kandydatów do filtrowania
            top_tokens = torch.topk(mask_token_logits, min(top_k * 5, 200), dim=0).indices.tolist()
            
            candidates = []
            all_words = []  # Wszystkie słowa jako fallback
            
            for token in top_tokens:
                word = self.tokenizer.decode([token]).strip()
                word = word.replace("##", "").strip()
                
                # Zbierz wszystkie alfabetyczne słowa
                if word and len(word) > 1 and word.replace("-", "").replace("'", "").isalpha():
                    all_words.append(word)
                    # Preferuj te spełniające filtr
                    if filter_func is None or filter_func(word):
                        candidates.append(word)
                        if len(candidates) >= top_k:
                            break
            
            # Użyj kandydatów lub wszystkich słów lub fallback
            word_list = candidates if candidates else (all_words[:top_k] if all_words else [fallback])
            word = random.choice(word_list)
            
            # Popraw kapitalizację
            return word[0].upper() + word[1:].lower() if len(word) > 1 else word.upper()
            
        except Exception as e:
            print(f"Błąd w predykcji Herbert: {e}")
            return fallback
    
    def generate_name_with_context(self, full_text: str, position: int) -> str:
        """Generuj imię z pełnym kontekstem tekstu."""
        def is_valid_name(word):
            return len(word) >= 3 and word[0].isupper() and not any(c.isdigit() for c in word)
        
        fallback = random.choice(["Anna", "Jan", "Maria", "Piotr", "Katarzyna", "Tomasz", "Andrzej"])
        return self.predict_with_context(full_text, position, window=150, top_k=20, 
                                        filter_func=is_valid_name, fallback=fallback)
    
    def generate_surname_with_context(self, full_text: str, position: int) -> str:
        """Generuj nazwisko z pełnym kontekstem tekstu."""
        def is_valid_surname(word):
            return len(word) >= 4 and word[0].isupper() and not any(c.isdigit() for c in word)
        
        fallback = random.choice(["Kowalski", "Nowak", "Wiśniewski", "Wójcik", "Kowalczyk", "Kamiński"])
        return self.predict_with_context(full_text, position, window=150, top_k=20,
                                        filter_func=is_valid_surname, fallback=fallback)
    
    def generate_sex_with_context(self, full_text: str, position: int) -> str:
        """Generuj płeć z pełnym kontekstem tekstu."""
        # Sprawdź czy w kontekście są wskazówki
        context_window = full_text[max(0, position-200):min(len(full_text), position+200)].lower()
        
        # Sprawdź imię w kontekście
        name_match = re.search(r'nazywam się (\w+)', context_window)
        if name_match:
            name = name_match.group(1).lower()
            # Typowe męskie końcówki
            if name.endswith(('ek', 'sław', 'mir', 'ciej', 'sz', 'rz')):
                return "mężczyzna"
            # Typowe damskie końcówki
            if name.endswith(('a', 'ia', 'ka', 'na')):
                return "kobieta"
        
        # Analiza kontekstu
        male_indicators = context_window.count('pan ') + context_window.count('mężczyzna') + context_window.count('chłopak')
        female_indicators = context_window.count('pani ') + context_window.count('kobieta') + context_window.count('dziewczyna')
        
        if male_indicators > female_indicators:
            return "mężczyzna"
        elif female_indicators > male_indicators:
            return "kobieta"
        
        # Ostatnia szansa - losuj na podstawie popularności
        return random.choice(["mężczyzna", "kobieta"])
    
    def generate_city_with_context(self, full_text: str, position: int) -> str:
        """Generuj miasto z pełnym kontekstem tekstu."""
        def is_valid_city(word):
            return len(word) >= 4 and word[0].isupper() and not any(c.isdigit() for c in word)
        
        fallback = random.choice(["Warszawa", "Kraków", "Gdańsk", "Wrocław", "Poznań", "Katowice"])
        return self.predict_with_context(full_text, position, window=150, top_k=25,
                                        filter_func=is_valid_city, fallback=fallback)
    
    def generate_company_with_context(self, full_text: str, position: int) -> str:
        """Generuj firmę z pełnym kontekstem tekstu."""
        def is_valid_company(word):
            return len(word) >= 3
        
        fallback = random.choice(["BudPol", "TechService", "ProfiGrup"])
        company_word = self.predict_with_context(full_text, position, window=150, top_k=15,
                                                 filter_func=is_valid_company, fallback=fallback)
        
        # Dodaj prefix jeśli go nie ma
        prefixes = ["FPUH", "PHU", "Gabinety", "Fundacja", "Przedsiębiorstwo", "Firma"]
        if not any(company_word.startswith(p) for p in prefixes):
            return f"{random.choice(prefixes)} {company_word}"
        return company_word
    
    def generate_name(self) -> str:
        """Generuj imię przy użyciu Herbert (fallback bez kontekstu)."""
        contexts = [
            "Nazywam się [MASK] Kowalski.",
            "Pan [MASK] Nowak pracuje w biurze.",
            "Pani [MASK] jest nauczycielską.",
            "Kolega [MASK] studiuje prawo.",
            "Imię: [MASK], nazwisko: Nowak."
        ]
        context = random.choice(contexts)
        
        def is_valid_name(word):
            return len(word) >= 3 and word[0].isupper() and not any(c.isdigit() for c in word)
        
        fallback = random.choice(["Anna", "Jan", "Maria", "Piotr", "Katarzyna", "Tomasz", "Andrzej"])
        return self.predict_with_herbert(context, top_k=20, filter_func=is_valid_name, fallback=fallback)
    
    def generate_surname(self) -> str:
        """Generuj nazwisko przy użyciu Herbert."""
        contexts = [
            "Pan Jan [MASK] przyszedł do urzędu.",
            "Pani Anna [MASK] pracuje w szkole.",
            "Inżynier [MASK] zaprojektował most.",
            "Nazwisko: [MASK], imię: Piotr.",
            "Doktor [MASK] przyjmuje w szpitalu."
        ]
        context = random.choice(contexts)
        
        def is_valid_surname(word):
            return len(word) >= 4 and word[0].isupper() and not any(c.isdigit() for c in word)
        
        fallback = random.choice(["Kowalski", "Nowak", "Wiśniewski", "Wójcik", "Kowalczyk", "Kamiński"])
        return self.predict_with_herbert(context, top_k=20, filter_func=is_valid_surname, fallback=fallback)
    
    def generate_city(self) -> str:
        """Generuj miasto przy użyciu Herbert."""
        contexts = [
            "Mieszkam w [MASK] od urodzenia.",
            "Jadę do [MASK] na spotkanie.",
            "Konferencja w [MASK] była udana.",
            "Uniwersytet w [MASK] jest bardzo dobry.",
            "Burmistrz [MASK] ogłosił decyzję."
        ]
        context = random.choice(contexts)
        
        def is_valid_city(word):
            return len(word) >= 4 and word[0].isupper() and not any(c.isdigit() for c in word)
        
        fallback = random.choice(["Warszawa", "Kraków", "Gdańsk", "Wrocław", "Poznań", "Katowice"])
        return self.predict_with_herbert(context, top_k=25, filter_func=is_valid_city, fallback=fallback)
    
    def generate_company(self) -> str:
        """Generuj nazwę firmy przy użyciu Herbert."""
        prefixes = ["FPUH", "PHU", "Gabinety", "Fundacja", "Przedsiębiorstwo", "Firma"]
        prefix = random.choice(prefixes)
        
        contexts = [
            f"{prefix} [MASK] świadczy usługi.",
            f"{prefix} [MASK] zatrudnia pracowników.",
            f"{prefix} [MASK] działa na rynku."
        ]
        context = random.choice(contexts)
        
        def is_valid_company(word):
            return len(word) >= 3
        
        company_name = self.predict_with_herbert(context, top_k=15, filter_func=is_valid_company)
        return f"{prefix} {company_name}"
    
    def generate_phone(self) -> str:
        """Generuj numer telefonu."""
        prefix = random.choice(["+48 ", "", ""])
        number = f"{random.randint(500, 799)} {random.randint(100, 999)} {random.randint(100, 999)}"
        return f"{prefix}{number}"
    
    def generate_email(self, context_name: str = None, context_surname: str = None) -> str:
        """Generuj adres email używając Herberta i kontekstu."""
        # Jeśli mamy kontekst (np. z otaczającego tekstu), użyj go
        if context_name and context_surname:
            name = context_name.lower()
            surname = context_surname.lower()
        else:
            # Generuj nowe imię i nazwisko
            name = self.generate_name().lower()
            surname = self.generate_surname().lower()
        
        domain = random.choice(["example.com", "example.net", "example.org"])
        
        patterns = [
            f"{name}{surname}",
            f"{name}.{surname}",
            f"{name[0]}{surname}",
            f"{surname}{random.randint(10, 99)}"
        ]
        
        pattern = random.choice(patterns)
        # Usuń polskie znaki
        trans = str.maketrans("ąćęłńóśźż", "acelnoszz")
        pattern = pattern.translate(trans)
        
        return f"{pattern}@{domain}"
    
    def generate_pesel(self) -> str:
        """Generuj PESEL."""
        year = random.randint(60, 99)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        serial = random.randint(100, 999)
        
        # Uproszczona generacja - bez pełnej walidacji
        pesel = f"{year:02d}{month:02d}{day:02d}{serial:03d}"
        # Dodaj cyfrę kontrolną (uproszczona)
        check_digit = random.randint(0, 9)
        
        return pesel + str(check_digit)
    
    def generate_document_number(self) -> str:
        """Generuj numer dokumentu."""
        letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
        numbers = ''.join(random.choices('0123456789', k=4))
        return f"{letters}{numbers}"
    
    def generate_age(self) -> str:
        """Generuj wiek."""
        return str(random.randint(18, 85))
    
    def generate_sex(self) -> str:
        """Generuj płeć przy użyciu Herbert."""
        contexts = [
            "Jestem [MASK] i pracuję w firmie.",
            "Jako [MASK] mam różne doświadczenia.",
            "Płeć: [MASK]."
        ]
        context = random.choice(contexts)
        result = self.predict_with_herbert(context, top_k=10).lower()
        
        if "kobieta" in result or "pani" in result:
            return "kobieta"
        elif "mężczyzna" in result or "pan" in result or "męski" in result:
            return "mężczyzna"
        return random.choice(["mężczyzna", "kobieta"])
    
    def generate_date(self) -> str:
        """Generuj datę."""
        start = datetime(2000, 1, 1)
        end = datetime(2024, 12, 31)
        days = (end - start).days
        random_date = start + timedelta(days=random.randrange(days))
        return random_date.strftime("%d %m %Y")
    
    def generate_address(self) -> str:
        """Generuj adres przy użyciu Herbert dla nazw ulic."""
        street_type = random.choice(self.street_types)
        
        contexts = [
            f"Mieszkam przy {street_type} [MASK] numer 15.",
            f"Sklep na {street_type} [MASK] jest czynny.",
            f"Adres: {street_type} [MASK] 42.",
            f"Biuro przy {street_type} [MASK] w centrum."
        ]
        context = random.choice(contexts)
        
        def is_valid_street(word):
            return len(word) >= 5 and word[0].isupper() and not any(c.isdigit() for c in word)
        
        fallback = random.choice(["Główna", "Kwiatowa", "Słoneczna", "Leśna", "Kolejowa", "Krzywa"])
        street_name = self.predict_with_herbert(context, top_k=25, filter_func=is_valid_street, fallback=fallback)
        number = random.randint(1, 200)
        
        if random.random() < 0.3:
            apartment = random.randint(1, 100)
            return f"{street_type} {street_name} {number}/{apartment}"
        return f"{street_type} {street_name} {number}"
    
    def generate_bank_account(self) -> str:
        """Generuj numer konta bankowego."""
        return ' '.join([''.join(random.choices('0123456789', k=4)) for _ in range(6)])
    
    def deanonymize_text(self, text: str) -> str:
        """Deanonimizuj tekst zamieniając tokeny na wygenerowane wartości z kontekstem.
        Obsługuje zarówno format [type] jak i [MASK:type]."""
        result = text
        
        # Śledź wygenerowane imiona/nazwiska dla spójności w obrębie akapitu
        paragraph_context = {}
        
        # Kolejność ma znaczenie - najpierw rzeczy używające kontekstu
        # Obsługa obu formatów: [type] i [MASK:type]
        all_patterns = [
            (r'\[(?:MASK:)?name\]', 'name'),
            (r'\[(?:MASK:)?surname\]', 'surname'),
            (r'\[(?:MASK:)?sex\]', 'sex'),
            (r'\[(?:MASK:)?city\]', 'city'),
            (r'\[(?:MASK:)?company\]', 'company'),
            (r'\[(?:MASK:)?address\]', 'address'),
            (r'\[(?:MASK:)?phone\]', 'phone'),
            (r'\[(?:MASK:)?age\]', 'age'),
            (r'\[(?:MASK:)?email\]', 'email'),
            (r'\[(?:MASK:)?pesel\]', 'pesel'),
            (r'\[(?:MASK:)?document-number\]', 'document'),
            (r'\[(?:MASK:)?date\]', 'date'),
            (r'\[(?:MASK:)?data\]', 'data'),
            (r'\[(?:MASK:)?bank-account\]', 'bank'),
        ]
        
        for pattern, token_type in all_patterns:
            matches = list(re.finditer(pattern, result, re.IGNORECASE))
            
            for match in reversed(matches):  # Od końca, żeby nie zepsuć indeksów
                # Zamień [token] na [MASK] tymczasowo
                temp_text = result[:match.start()] + "[MASK]" + result[match.end():]
                
                # Generuj z pełnym kontekstem
                if token_type == 'name':
                    replacement = self.generate_name_with_context(temp_text, match.start())
                    paragraph_context['last_name'] = replacement
                elif token_type == 'surname':
                    replacement = self.generate_surname_with_context(temp_text, match.start())
                    paragraph_context['last_surname'] = replacement
                elif token_type == 'sex':
                    replacement = self.generate_sex_with_context(temp_text, match.start())
                elif token_type == 'city':
                    replacement = self.generate_city_with_context(temp_text, match.start())
                elif token_type == 'company':
                    replacement = self.generate_company_with_context(temp_text, match.start())
                elif token_type == 'address':
                    replacement = self.generate_address()
                elif token_type == 'email':
                    # Szukaj imienia i nazwiska w pobli\u017cu
                    context_before = result[max(0, match.start()-200):match.start()]
                    name_match = re.search(r'Nazywam si\u0119 (\w+) (\w+)', context_before)
                    if name_match:
                        replacement = self.generate_email(name_match.group(1), name_match.group(2))
                    else:
                        replacement = self.generate_email(
                            paragraph_context.get('last_name'),
                            paragraph_context.get('last_surname')
                        )
                elif token_type == 'phone':
                    replacement = self.generate_phone()
                elif token_type == 'age':
                    replacement = self.generate_age()
                elif token_type == 'pesel':
                    replacement = self.generate_pesel()
                elif token_type == 'document':
                    replacement = self.generate_document_number()
                elif token_type in ['date', 'data']:
                    replacement = self.generate_date()
                elif token_type == 'bank':
                    replacement = self.generate_bank_account()
                else:
                    replacement = "???"
                
                result = result[:match.start()] + replacement + result[match.end():]
        
        return result
    
    def process_file(self, input_path: str, output_path: str):
        """Przetwórz plik tekstowy."""
        print(f"Czytam plik: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print("Deanonimizuję tekst...")
        deanonymized = self.deanonymize_text(text)
        
        print(f"Zapisuję wynik do: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(deanonymized)
        
        print("Gotowe!")


def main():
    parser = argparse.ArgumentParser(description='Deanonimizacja tekstu przy użyciu modelu Herbert')
    parser.add_argument('input_file', help='Plik wejściowy z tokenami')
    parser.add_argument('output_file', help='Plik wyjściowy z deanonimizowanym tekstem')
    parser.add_argument('--model', default='allegro/herbert-base-cased', 
                       help='Nazwa modelu Herbert (domyślnie: allegro/herbert-base-cased)')
    
    args = parser.parse_args()
    
    # Sprawdź czy lokalny model istnieje i ma niezbędne pliki
    from pathlib import Path
    if not args.model.startswith('allegro/') and not args.model.startswith('huggingface/'):
        model_path = Path(args.model)
        if model_path.exists():
            required_files = ['config.json', 'pytorch_model.bin']
            tokenizer_files = ['tokenizer_config.json', 'vocab.txt']
            
            has_model = any((model_path / f).exists() for f in required_files)
            has_tokenizer = any((model_path / f).exists() for f in tokenizer_files)
            
            if not has_model or not has_tokenizer:
                print(f"\n⚠️  UWAGA: Katalog {args.model} nie zawiera kompletnego modelu!")
                print("\nModel może nie być jeszcze wytrenowany. Aby wytrenować model uruchom:")
                print("  ./train.sh")
                print("\nLub użyj bazowego modelu Herbert:")
                print("  python src/scripts/deanonymize_herbert.py input.txt output.txt")
                print("\nKontynuuję z bazowym modelem...\n")
                args.model = 'allegro/herbert-base-cased'
        else:
            print(f"\n⚠️  UWAGA: Katalog {args.model} nie istnieje!")
            print("\nUżywam bazowego modelu Herbert: allegro/herbert-base-cased\n")
            args.model = 'allegro/herbert-base-cased'
    
    deanonymizer = HerbertDeanonymizer(model_name=args.model)
    deanonymizer.process_file(args.input_file, args.output_file)


if __name__ == '__main__':
    main()
