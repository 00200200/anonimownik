"""
Skrypt do deanonimizacji tekstu generując syntetyczne dane osobowe.
Używa losowych wartości dla imion, nazwisk, miast oraz generuje PESEL, telefony, etc.
"""
import re
import random
from typing import Dict, List, Tuple

class Deanonymizer:
    def __init__(self):
        """Inicjalizacja generatorów danych"""
        print("Inicjalizacja generatora danych...")
        
        # Listy do losowania
        self.polish_first_names_male = [
            "Jan", "Piotr", "Krzysztof", "Andrzej", "Tomasz", "Paweł", "Michał",
            "Stanisław", "Marek", "Marcin", "Jacek", "Wojciech", "Grzegorz",
            "Zbigniew", "Józef", "Adam", "Dariusz", "Bartosz", "Mateusz", "Łukasz",
            "Kamil", "Rafał", "Robert", "Daniel", "Dawid", "Sebastian", "Artur"
        ]
        
        self.polish_first_names_female = [
            "Maria", "Anna", "Katarzyna", "Małgorzata", "Agnieszka", "Barbara",
            "Ewa", "Krystyna", "Elżbieta", "Joanna", "Magdalena", "Monika",
            "Zofia", "Teresa", "Jadwiga", "Danuta", "Halina", "Irena", "Natalia",
            "Aleksandra", "Beata", "Iwona", "Dorota", "Karolina", "Paulina"
        ]
        
        self.polish_surnames = [
            "Nowak", "Kowalski", "Wiśniewski", "Wójcik", "Kowalczyk", "Kamiński",
            "Lewandowski", "Zieliński", "Szymański", "Woźniak", "Dąbrowski",
            "Kozłowski", "Jankowski", "Mazur", "Kwiatkowski", "Krawczyk", "Piotrowski",
            "Grabowski", "Nowakowski", "Pawłowski", "Michalski", "Nowicki", "Adamczyk",
            "Dudek", "Zajączkowski", "Król", "Sikora", "Baran", "Rutkowski", "Michalak"
        ]
        
        self.polish_cities = [
            "Warszawa", "Kraków", "Łódź", "Wrocław", "Poznań", "Gdańsk", "Szczecin",
            "Bydgoszcz", "Lublin", "Katowice", "Białystok", "Gdynia", "Częstochowa",
            "Radom", "Sosnowiec", "Toruń", "Kielce", "Gliwice", "Zabrze", "Bytom",
            "Olsztyn", "Bielsko-Biała", "Rzeszów", "Ruda Śląska", "Rybnik", "Tychy",
            "Dąbrowa Górnicza", "Płock", "Elbląg", "Opole", "Gorzów Wielkopolski"
        ]
        
        self.street_names = [
            "Główna", "Polna", "Słoneczna", "Krótka", "Ogrodowa", "Leśna", "Spacerowa",
            "Parkowa", "Lipowa", "Klonowa", "Warszawska", "Kwiatowa", "Mickiewicza",
            "Kościuszki", "Słowackiego", "Piłsudskiego", "Sienkiewicza", "Chopina"
        ]
        
        self.street_types = ["ul.", "al.", "plac", "os."]
        
        self.companies = [
            "Gabinety", "FPUH", "Fundacja", "Przedsiębiorstwo", "Biuro", "Kancelaria",
            "Zakład", "Firma", "Usługi", "PHU", "Spółka", "Instytut"
        ]
        
        self.company_suffixes = [
            "Sp. z o.o.", "S.A.", "i Wspólnicy", "& Partners", "Group", "Service"
        ]
        
    def generate_pesel(self) -> str:
        """Generuje losowy, poprawny numer PESEL"""
        # Rok (00-99)
        year = random.randint(40, 99)  # 1940-1999
        # Miesiąc (01-12)
        month = random.randint(1, 12)
        # Dzień (01-28 dla uproszczenia)
        day = random.randint(1, 28)
        # Numer serii (3 cyfry)
        series = random.randint(0, 999)
        # Płeć (parzysta=kobieta, nieparzysta=mężczyzna)
        sex_digit = random.randint(0, 9)
        
        pesel_base = f"{year:02d}{month:02d}{day:02d}{series:03d}{sex_digit}"
        
        # Oblicz cyfrę kontrolną
        weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
        checksum = sum(int(pesel_base[i]) * weights[i] for i in range(10)) % 10
        checksum = (10 - checksum) % 10
        
        return pesel_base + str(checksum)
    
    def generate_phone(self) -> str:
        """Generuje losowy polski numer telefonu"""
        prefixes = ["48", "+48", ""]
        prefix = random.choice(prefixes)
        
        # Popularne prefiksy operatorów
        operator_codes = ["50", "51", "53", "57", "60", "66", "69", "72", "73", "78", "79", "88"]
        operator = random.choice(operator_codes)
        
        number = f"{random.randint(0, 999):03d} {random.randint(0, 999):03d}"
        
        if prefix:
            return f"{prefix} {operator} {number}"
        return f"{operator}{number.replace(' ', ' ')}"
    
    def generate_email(self, name: str = None, surname: str = None) -> str:
        """Generuje losowy adres email"""
        if name and surname:
            base = f"{name.lower()}{surname.lower()}"
        else:
            base = f"user{random.randint(1000, 9999)}"
        
        domains = ["example.com", "example.org", "example.net", "mail.pl", "gmail.com"]
        separators = ["", ".", "_", str(random.randint(0, 99))]
        
        return f"{base}{random.choice(separators)}@{random.choice(domains)}"
    
    def generate_address(self) -> str:
        """Generuje losowy adres"""
        street_type = random.choice(self.street_types)
        street_name = random.choice(self.street_names)
        
        # Numer budynku
        building = random.randint(1, 500)
        
        # Czasem dodaj numer mieszkania
        if random.random() > 0.5:
            apartment = random.randint(1, 100)
            address = f"{street_type} {street_name} {building}/{apartment}"
        else:
            address = f"{street_type} {street_name} {building}"
        
        # Kod pocztowy i miasto
        postal_code = f"{random.randint(10, 99)}-{random.randint(100, 999)}"
        city = random.choice(self.polish_cities)
        
        return f"{address} {postal_code} {city}"
    
    def generate_document_number(self) -> str:
        """Generuje losowy numer dowodu osobistego"""
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return f"{random.choice(letters)}{random.choice(letters)}{random.randint(0, 9)}{random.randint(0, 9)}{random.randint(0, 9)}{random.randint(0, 9)}"
    
    def generate_date(self) -> str:
        """Generuje losową datę"""
        day = random.randint(1, 28)
        month = random.choice(["stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca", 
                              "lipca", "sierpnia", "września", "października", "listopada", "grudnia"])
        year = random.randint(1970, 2023)
        return f"{day} {month} {year}"
    
    def generate_company_name(self) -> str:
        """Generuje losową nazwę firmy"""
        base = random.choice(self.companies)
        
        if random.random() > 0.5:
            # Dodaj nazwisko właściciela
            surname = random.choice(self.polish_surnames)
            name = f"{base} {surname}"
        else:
            # Dodaj wymyśloną nazwę
            word = random.choice(["Tech", "Pro", "Expert", "Master", "Global", "Consulting"])
            name = f"{base} {word}"
        
        if random.random() > 0.7:
            name += f" {random.choice(self.company_suffixes)}"
        
        return name
    
    def generate_bank_account(self) -> str:
        """Generuje losowy numer konta bankowego"""
        # Format: XXXX XXXX XXXX XXXX XXXX XXXX
        parts = [f"{random.randint(0, 9999):04d}" for _ in range(6)]
        return " ".join(parts)
    

    def replace_tokens(self, text: str) -> str:
        """Zamienia wszystkie tokeny w tekście, obsługuje zarówno [type] jak i [MASK:type]"""
        result = text
        
        # Słownik do przechowywania wygenerowanych wartości dla spójności
        generated_values = {}
        
        # Najpierw znajdź wszystkie tokeny - obsługa obu formatów: [type] i [MASK:type]
        tokens = re.findall(r'\[(?:MASK:)?([\w\-]+)\]', text)
        
        # Wygeneruj płeć i wiek najpierw (mogą być używane wielokrotnie)
        if 'sex' in tokens:
            sex = random.choice(['mężczyzna', 'kobieta'])
            generated_values['sex'] = sex
        
        if 'age' in tokens:
            age = random.randint(18, 90)
            generated_values['age'] = str(age)
        
        # Generuj imiona i nazwiska (mogą powtarzać się w tekście)
        if 'name' in tokens:
            # Dla małych fragmentów kontekstu użyj generatora
            sex = generated_values.get('sex', random.choice(['mężczyzna', 'kobieta']))
            if sex == 'mężczyzna' or 'mężczyzna' in text[:200]:
                name = random.choice(self.polish_first_names_male)
            else:
                name = random.choice(self.polish_first_names_female)
            generated_values['name'] = name
        
        if 'surname' in tokens:
            surname = random.choice(self.polish_surnames)
            generated_values['surname'] = surname
        
        if 'city' in tokens:
            city = random.choice(self.polish_cities)
            generated_values['city'] = city
        
        if 'address' in tokens:
            address = self.generate_address()
            generated_values['address'] = address
        
        if 'phone' in tokens:
            phone = self.generate_phone()
            generated_values['phone'] = phone
        
        if 'email' in tokens:
            name = generated_values.get('name', '')
            surname = generated_values.get('surname', '')
            email = self.generate_email(name, surname)
            generated_values['email'] = email
        
        if 'pesel' in tokens:
            pesel = self.generate_pesel()
            generated_values['pesel'] = pesel
        
        if 'document-number' in tokens:
            doc_num = self.generate_document_number()
            generated_values['document-number'] = doc_num
        
        if 'company' in tokens:
            company = self.generate_company_name()
            generated_values['company'] = company
        
        if 'date' in tokens or 'data' in tokens:
            date = self.generate_date()
            generated_values['date'] = date
            generated_values['data'] = date
        
        if 'bank-account' in tokens:
            account = self.generate_bank_account()
            generated_values['bank-account'] = account
        
        # Zamień wszystkie tokeny - obsługa obu formatów: [type] i [MASK:type]
        for token, value in generated_values.items():
            result = result.replace(f'[{token}]', value)
            result = result.replace(f'[MASK:{token}]', value)
        
        return result
    
    def process_file(self, input_path: str, output_path: str):
        """Przetwarza cały plik"""
        print(f"Czytanie pliku: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Znaleziono {len(text)} znaków")
        
        # Podziel na linie dla lepszej kontroli
        lines = text.split('\n')
        processed_lines = []
        
        print(f"Przetwarzanie {len(lines)} linii...")
        for i, line in enumerate(lines):
            if i % 100 == 0:
                print(f"Przetworzono {i}/{len(lines)} linii...")
            
            processed_line = self.replace_tokens(line)
            processed_lines.append(processed_line)
        
        result = '\n'.join(processed_lines)
        
        print(f"Zapisywanie do: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        
        print("Gotowe!")


def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Użycie: python deanonymize.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    deanonymizer = Deanonymizer()
    deanonymizer.process_file(input_file, output_file)


if __name__ == "__main__":
    main()
