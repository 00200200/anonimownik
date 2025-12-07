
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import spacy

NAMES_FEMALE = {
    "Anna": {"nom": "Anna", "gen": "Anny", "dat": "Annie", "acc": "Annę", "inst": "Anną", "loc": "Annie"},
    "Maria": {"nom": "Maria", "gen": "Marii", "dat": "Marii", "acc": "Marię", "inst": "Marią", "loc": "Marii"},
    "Katarzyna": {"nom": "Katarzyna", "gen": "Katarzyny", "dat": "Katarzynie", "acc": "Katarzynę", "inst": "Katarzyną", "loc": "Katarzynie"},
    "Agnieszka": {"nom": "Agnieszka", "gen": "Agnieszki", "dat": "Agnieszce", "acc": "Agnieszkę", "inst": "Agnieszką", "loc": "Agnieszce"},
    "Magdalena": {"nom": "Magdalena", "gen": "Magdaleny", "dat": "Magdalenie", "acc": "Magdalenę", "inst": "Magdaleną", "loc": "Magdalenie"},
    "Ewa": {"nom": "Ewa", "gen": "Ewy", "dat": "Ewie", "acc": "Ewę", "inst": "Ewą", "loc": "Ewie"},
    "Joanna": {"nom": "Joanna", "gen": "Joanny", "dat": "Joannie", "acc": "Joannę", "inst": "Joanną", "loc": "Joannie"},
    "Monika": {"nom": "Monika", "gen": "Moniki", "dat": "Monice", "acc": "Monikę", "inst": "Moniką", "loc": "Monice"},
    "Natalia": {"nom": "Natalia", "gen": "Natalii", "dat": "Natalii", "acc": "Natalię", "inst": "Natalią", "loc": "Natalii"},
    "Karolina": {"nom": "Karolina", "gen": "Karoliny", "dat": "Karolinie", "acc": "Karolinę", "inst": "Karoliną", "loc": "Karolinie"},
    "Aleksandra": {"nom": "Aleksandra", "gen": "Aleksandry", "dat": "Aleksandrze", "acc": "Aleksandrę", "inst": "Aleksandrą", "loc": "Aleksandrze"},
    "Paulina": {"nom": "Paulina", "gen": "Pauliny", "dat": "Paulinie", "acc": "Paulinę", "inst": "Pauliną", "loc": "Paulinie"},
    "Weronika": {"nom": "Weronika", "gen": "Weroniki", "dat": "Weronice", "acc": "Weronikę", "inst": "Weroniką", "loc": "Weronice"},
    "Zofia": {"nom": "Zofia", "gen": "Zofii", "dat": "Zofii", "acc": "Zofię", "inst": "Zofią", "loc": "Zofii"},
    "Julia": {"nom": "Julia", "gen": "Julii", "dat": "Julii", "acc": "Julię", "inst": "Julią", "loc": "Julii"},
}

# Imiona męskie z formami odmienionymi
NAMES_MALE = {
    "Jan": {"nom": "Jan", "gen": "Jana", "dat": "Janowi", "acc": "Jana", "inst": "Janem", "loc": "Janie"},
    "Piotr": {"nom": "Piotr", "gen": "Piotra", "dat": "Piotrowi", "acc": "Piotra", "inst": "Piotrem", "loc": "Piotrze"},
    "Michał": {"nom": "Michał", "gen": "Michała", "dat": "Michałowi", "acc": "Michała", "inst": "Michałem", "loc": "Michale"},
    "Tomasz": {"nom": "Tomasz", "gen": "Tomasza", "dat": "Tomaszowi", "acc": "Tomasza", "inst": "Tomaszem", "loc": "Tomaszu"},
    "Krzysztof": {"nom": "Krzysztof", "gen": "Krzysztofa", "dat": "Krzysztofowi", "acc": "Krzysztofa", "inst": "Krzysztofem", "loc": "Krzysztofie"},
    "Paweł": {"nom": "Paweł", "gen": "Pawła", "dat": "Pawłowi", "acc": "Pawła", "inst": "Pawłem", "loc": "Pawle"},
    "Marcin": {"nom": "Marcin", "gen": "Marcina", "dat": "Marcinowi", "acc": "Marcina", "inst": "Marcinem", "loc": "Marcinie"},
    "Łukasz": {"nom": "Łukasz", "gen": "Łukasza", "dat": "Łukaszowi", "acc": "Łukasza", "inst": "Łukaszem", "loc": "Łukaszu"},
    "Jakub": {"nom": "Jakub", "gen": "Jakuba", "dat": "Jakubowi", "acc": "Jakuba", "inst": "Jakubem", "loc": "Jakubie"},
    "Marek": {"nom": "Marek", "gen": "Marka", "dat": "Markowi", "acc": "Marka", "inst": "Markiem", "loc": "Marku"},
    "Bartosz": {"nom": "Bartosz", "gen": "Bartosza", "dat": "Bartoszowi", "acc": "Bartosza", "inst": "Bartoszem", "loc": "Bartoszu"},
    "Dawid": {"nom": "Dawid", "gen": "Dawida", "dat": "Dawidowi", "acc": "Dawida", "inst": "Dawidem", "loc": "Dawidzie"},
    "Adam": {"nom": "Adam", "gen": "Adama", "dat": "Adamowi", "acc": "Adama", "inst": "Adamem", "loc": "Adamie"},
    "Kamil": {"nom": "Kamil", "gen": "Kamila", "dat": "Kamilowi", "acc": "Kamila", "inst": "Kamilem", "loc": "Kamilu"},
}

# Nazwiska żeńskie
SURNAMES_FEMALE = {
    "Nowak": {"nom": "Nowak", "gen": "Nowak", "dat": "Nowak", "acc": "Nowak", "inst": "Nowak", "loc": "Nowak"},
    "Kowalska": {"nom": "Kowalska", "gen": "Kowalskiej", "dat": "Kowalskiej", "acc": "Kowalską", "inst": "Kowalską", "loc": "Kowalskiej"},
    "Wiśniewska": {"nom": "Wiśniewska", "gen": "Wiśniewskiej", "dat": "Wiśniewskiej", "acc": "Wiśniewską", "inst": "Wiśniewską", "loc": "Wiśniewskiej"},
    "Zielińska": {"nom": "Zielińska", "gen": "Zielińskiej", "dat": "Zielińskiej", "acc": "Zielińską", "inst": "Zielińską", "loc": "Zielińskiej"},
    "Kamińska": {"nom": "Kamińska", "gen": "Kamińskiej", "dat": "Kamińskiej", "acc": "Kamińską", "inst": "Kamińską", "loc": "Kamińskiej"},
    "Lewandowska": {"nom": "Lewandowska", "gen": "Lewandowskiej", "dat": "Lewandowskiej", "acc": "Lewandowską", "inst": "Lewandowską", "loc": "Lewandowskiej"},
    "Szymańska": {"nom": "Szymańska", "gen": "Szymańskiej", "dat": "Szymańskiej", "acc": "Szymańską", "inst": "Szymańską", "loc": "Szymańskiej"},
    "Wójcik": {"nom": "Wójcik", "gen": "Wójcik", "dat": "Wójcik", "acc": "Wójcik", "inst": "Wójcik", "loc": "Wójcik"},
}

# Nazwiska męskie
SURNAMES_MALE = {
    "Nowak": {"nom": "Nowak", "gen": "Nowaka", "dat": "Nowakowi", "acc": "Nowaka", "inst": "Nowakiem", "loc": "Nowaku"},
    "Kowalski": {"nom": "Kowalski", "gen": "Kowalskiego", "dat": "Kowalskiemu", "acc": "Kowalskiego", "inst": "Kowalskim", "loc": "Kowalskim"},
    "Wiśniewski": {"nom": "Wiśniewski", "gen": "Wiśniewskiego", "dat": "Wiśniewskiemu", "acc": "Wiśniewskiego", "inst": "Wiśniewskim", "loc": "Wiśniewskim"},
    "Zieliński": {"nom": "Zieliński", "gen": "Zielińskiego", "dat": "Zielińskiemu", "acc": "Zielińskiego", "inst": "Zielińskim", "loc": "Zielińskim"},
    "Kamiński": {"nom": "Kamiński", "gen": "Kamińskiego", "dat": "Kamińskiemu", "acc": "Kamińskiego", "inst": "Kamińskim", "loc": "Kamińskim"},
    "Lewandowski": {"nom": "Lewandowski", "gen": "Lewandowskiego", "dat": "Lewandowskiemu", "acc": "Lewandowskiego", "inst": "Lewandowskim", "loc": "Lewandowskim"},
    "Szymański": {"nom": "Szymański", "gen": "Szymańskiego", "dat": "Szymańskiemu", "acc": "Szymańskiego", "inst": "Szymańskim", "loc": "Szymańskim"},
    "Wójcik": {"nom": "Wójcik", "gen": "Wójcika", "dat": "Wójcikowi", "acc": "Wójcika", "inst": "Wójcikiem", "loc": "Wójciku"},
}

# Miasta z odmianami
CITIES = {
    "Warszawa": {"nom": "Warszawa", "gen": "Warszawy", "dat": "Warszawie", "acc": "Warszawę", "inst": "Warszawą", "loc": "Warszawie"},
    "Kraków": {"nom": "Kraków", "gen": "Krakowa", "dat": "Krakowowi", "acc": "Kraków", "inst": "Krakowem", "loc": "Krakowie"},
    "Gdańsk": {"nom": "Gdańsk", "gen": "Gdańska", "dat": "Gdańskowi", "acc": "Gdańsk", "inst": "Gdańskiem", "loc": "Gdańsku"},
    "Wrocław": {"nom": "Wrocław", "gen": "Wrocławia", "dat": "Wrocławiowi", "acc": "Wrocław", "inst": "Wrocławiem", "loc": "Wrocławiu"},
    "Poznań": {"nom": "Poznań", "gen": "Poznania", "dat": "Poznaniowi", "acc": "Poznań", "inst": "Poznaniem", "loc": "Poznaniu"},
    "Łódź": {"nom": "Łódź", "gen": "Łodzi", "dat": "Łodzi", "acc": "Łódź", "inst": "Łodzią", "loc": "Łodzi"},
    "Katowice": {"nom": "Katowice", "gen": "Katowic", "dat": "Katowicom", "acc": "Katowice", "inst": "Katowicami", "loc": "Katowicach"},
    "Lublin": {"nom": "Lublin", "gen": "Lublina", "dat": "Lublinowi", "acc": "Lublin", "inst": "Lublinem", "loc": "Lublinie"},
    "Białystok": {"nom": "Białystok", "gen": "Białegostoku", "dat": "Białemustokowi", "acc": "Białystok", "inst": "Białymstokiem", "loc": "Białymstoku"},
    "Szczecin": {"nom": "Szczecin", "gen": "Szczecina", "dat": "Szczecinowi", "acc": "Szczecin", "inst": "Szczecinem", "loc": "Szczecinie"},
}

# Poglądy polityczne (nieodmienne lub z prostą odmianą)
POLITICAL_VIEWS = [
    "PiS", "Platforma Obywatelska", "Konfederacja", "Lewica", "Polska 2050",
    "PSL", "Nowa Lewica", "Razem"
]

# Firmy (nieodmienne)
COMPANIES = [
    "CD Projekt", "Allegro", "InPost", "PKO BP", "Orlen", "PZU", "KGHM",
    "Orange", "T-Mobile", "Play", "Asseco", "Comarch", "LPP", "CCC"
]

# Stanowiska pracy
JOB_TITLES = {
    "programista": {"nom": "programista", "gen": "programisty", "dat": "programiście", "acc": "programistę", "inst": "programistą", "loc": "programiście"},
    "analityk": {"nom": "analityk", "gen": "analityka", "dat": "analitykowi", "acc": "analityka", "inst": "analitykiem", "loc": "analityku"},
    "kierownik": {"nom": "kierownik", "gen": "kierownika", "dat": "kierownikowi", "acc": "kierownika", "inst": "kierownikiem", "loc": "kierowniku"},
    "tester": {"nom": "tester", "gen": "testera", "dat": "testerowi", "acc": "testera", "inst": "testerem", "loc": "testerze"},
    "architekt": {"nom": "architekt", "gen": "architekta", "dat": "architektowi", "acc": "architekta", "inst": "architektem", "loc": "architekcie"},
    "specjalista": {"nom": "specjalista", "gen": "specjalisty", "dat": "specjaliście", "acc": "specjalistę", "inst": "specjalistą", "loc": "specjaliście"},
    "dyrektor": {"nom": "dyrektor", "gen": "dyrektora", "dat": "dyrektorowi", "acc": "dyrektora", "inst": "dyrektorem", "loc": "dyrektorze"},
}

# Szkoły (nieodmienne lub z prostą odmianą)
SCHOOLS = [
    "Politechnika Warszawska", "AGH", "Uniwersytet Warszawski", "Uniwersytet Jagielloński",
    "Politechnika Gdańska", "SGH", "UAM", "Uniwersytet Wrocławski"
]

# Adresy (nieodmienne)
ADDRESSES = [
    "ul. Długa 5", "ul. Prosta 12/3", "ul. Krótka 8", "ul. Główna 15",
    "ul. Nowa 22", "ul. Stara 7", "ul. Zielona 10", "ul. Niebieska 3"
]


# Mapowanie przypadków spaCy na skróty
CASE_MAPPING = {
    "Nom": "nom",
    "Gen": "gen",
    "Dat": "dat",
    "Acc": "acc",
    "Ins": "inst",
    "Loc": "loc",
    "Voc": "nom",  # wołacz -> mianownik dla uproszczenia
}


class SyntheticGenerator:
    """Generator syntetyczny - zamienia tokeny na nowe wartości z odmianą morfologiczną."""
    
    def __init__(self, seed: Optional[int] = None, spacy_model: str = "pl_core_news_sm"):
        """
        Args:
            seed: Ziarno losowości dla powtarzalności wyników
            spacy_model: Model spaCy do analizy morfologicznej
        """
        if seed is not None:
            random.seed(seed)
        
        # Załaduj spaCy
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            self.nlp = None
        
        # Słowniki z odmianami
        self.names_female = NAMES_FEMALE.copy()
        self.names_male = NAMES_MALE.copy()
        self.surnames_female = SURNAMES_FEMALE.copy()
        self.surnames_male = SURNAMES_MALE.copy()
        self.cities = CITIES.copy()
        self.job_titles = JOB_TITLES.copy()
        
        # Proste słowniki (nieodmienne)
        self.political_views = POLITICAL_VIEWS.copy()
        self.companies = COMPANIES.copy()
        self.schools = SCHOOLS.copy()
        self.addresses = ADDRESSES.copy()
    
    def add_entity_dictionary(self, entity_type: str, values: List[str]) -> None:
        """
        Dodaje lub nadpisuje słownik dla danego typu encji.
        
        Uwaga: Dla typów z odmianą (name, surname, city, job-title) wartości 
        będą używane tylko w mianowniku. Dla pełnej odmiany użyj słowników
        z wszystkimi przypadkami.
        
        Args:
            entity_type: Typ encji (np. "name", "city", "company")
            values: Lista wartości w mianowniku
        """
        if entity_type == 'political-view':
            self.political_views = values
        elif entity_type == 'company':
            self.companies = values
        elif entity_type == 'school-name':
            self.schools = values
        elif entity_type == 'address':
            self.addresses = values
        elif entity_type == 'name':
            # Dla imion twórz prosty słownik z tylko mianownikiem
            self.names_male = {v: {"nom": v} for v in values}
            self.names_female = {}
        elif entity_type == 'surname':
            self.surnames_male = {v: {"nom": v} for v in values}
            self.surnames_female = {}
        elif entity_type == 'city':
            self.cities = {v: {"nom": v} for v in values}
        elif entity_type == 'job-title':
            self.job_titles = {v: {"nom": v} for v in values}
    
    def _detect_gender_from_text(self, text: str) -> str:
        """Wykrywa płeć na podstawie tekstu (końcówki)."""
        text_lower = text.lower()
        
        # Końcówki żeńskie
        if text_lower.endswith(('a', 'ej', 'ą', 'ę', 'ie', 'ii', 'i', 'y')):
            # Sprawdź czy to nie męskie nazwisko typu Nowak
            if text_lower.endswith(('ski', 'cki', 'wicz', 'ek', 'ik', 'ak')):
                return "masc"
            return "fem"
        
        return "masc"
    
    def _detect_case_from_morph(self, morph_features: Dict[str, str]) -> str:
        """Wykrywa przypadek z cech morfologicznych."""
        case = morph_features.get('case', 'Nom')
        return CASE_MAPPING.get(case, 'nom')
    
    def _detect_case_from_text(self, original_text: str) -> str:
        """Próbuje wykryć przypadek na podstawie końcówki tekstu."""
        if not self.nlp:
            return 'nom'
        
        # Analizuj tekst za pomocą spaCy
        doc = self.nlp(original_text)
        for token in doc:
            morph = token.morph.to_dict()
            if 'Case' in morph:
                return CASE_MAPPING.get(morph['Case'], 'nom')
        
        return 'nom'
    
    def _get_inflected_name(
        self, 
        gender: str, 
        case: str,
        original_text: Optional[str] = None
    ) -> str:
        """Pobiera odmienione imię."""
        if gender == "fem":
            name_dict = random.choice(list(self.names_female.values()))
        else:
            name_dict = random.choice(list(self.names_male.values()))
        
        return name_dict.get(case, name_dict.get('nom', list(name_dict.values())[0]))
    
    def _get_inflected_surname(
        self,
        gender: str,
        case: str,
        original_text: Optional[str] = None
    ) -> str:
        """Pobiera odmienione nazwisko."""
        if gender == "fem":
            surname_dict = random.choice(list(self.surnames_female.values()))
        else:
            surname_dict = random.choice(list(self.surnames_male.values()))
        
        return surname_dict.get(case, surname_dict.get('nom', list(surname_dict.values())[0]))
    
    def _get_inflected_city(self, case: str) -> str:
        """Pobiera odmienione miasto."""
        city_dict = random.choice(list(self.cities.values()))
        return city_dict.get(case, city_dict.get('nom', list(city_dict.values())[0]))
    
    def _get_inflected_job_title(self, case: str) -> str:
        """Pobiera odmienione stanowisko."""
        job_dict = random.choice(list(self.job_titles.values()))
        return job_dict.get(case, job_dict.get('nom', list(job_dict.values())[0]))
    
    def _get_value_for_entity(
        self,
        entity_type: str,
        morph_features: Dict[str, str],
        original_text: str
    ) -> Optional[str]:
        """
        Pobiera odpowiednio odmienioną wartość dla typu encji.
        
        Args:
            entity_type: Typ encji (name, surname, city, etc.)
            morph_features: Cechy morfologiczne
            original_text: Oryginalny tekst encji
        
        Returns:
            Odmieniona wartość lub None
        """
        # Wykryj przypadek
        if morph_features:
            case = self._detect_case_from_morph(morph_features)
        else:
            case = self._detect_case_from_text(original_text)
        
        # Wykryj płeć
        gender = morph_features.get('gender', '').lower()
        if not gender or gender not in ('fem', 'masc'):
            gender = self._detect_gender_from_text(original_text)
        
        # Mapowanie gender
        if gender in ('Fem', 'fem', 'f'):
            gender = 'fem'
        else:
            gender = 'masc'
        
        # Pobierz wartość w zależności od typu
        if entity_type == 'name':
            return self._get_inflected_name(gender, case, original_text)
        elif entity_type == 'surname':
            return self._get_inflected_surname(gender, case, original_text)
        elif entity_type == 'city':
            return self._get_inflected_city(case)
        elif entity_type == 'job-title':
            return self._get_inflected_job_title(case)
        elif entity_type == 'political-view':
            return random.choice(self.political_views)
        elif entity_type == 'company':
            return random.choice(self.companies)
        elif entity_type == 'school-name':
            return random.choice(self.schools)
        elif entity_type == 'address':
            return random.choice(self.addresses)
        elif entity_type == 'email':
            name = random.choice(list(self.names_male.keys()) + list(self.names_female.keys())).lower()
            surname = random.choice(list(self.surnames_male.keys())).lower()
            domain = random.choice(['gmail.com', 'wp.pl', 'onet.pl', 'o2.pl'])
            return f"{name}.{surname}@{domain}"
        elif entity_type == 'phone':
            return f"+48 {random.randint(500,899)} {random.randint(100,999)} {random.randint(100,999)}"
        elif entity_type == 'pesel':
            return ''.join([str(random.randint(0, 9)) for _ in range(11)])
        elif entity_type == 'bank-account':
            return ' '.join([f"{random.randint(10,99)}" if i == 0 else f"{random.randint(1000,9999)}" for i in range(7)])
        
        return None
    
    def generate(
        self, 
        anon_text: str, 
        entities_meta: List[Union[Dict, 'EntityMeta']]
    ) -> str:
        """
        Generuje tekst z nowymi, syntetycznymi danymi.
        
        Args:
            anon_text: Tekst z tokenami typu {name}, {surname} itp.
            entities_meta: Lista metadanych encji z modułu anonimizacji
        
        Returns:
            Tekst z podmienionymi tokenami na nowe wartości
        """
        result_text = anon_text
        
        # Przetwarzaj encje w kolejności występowania tokenów
        for entity in entities_meta:
            # Obsługa różnych formatów danych
            if hasattr(entity, 'entity_type'):
                entity_type = entity.entity_type
                original_text = entity.original_text
                morph_features = entity.morph_features
            else:
                entity_type = entity.get('entity_type') or entity.get('label')
                original_text = entity.get('original_text') or entity.get('value', '')
                morph_features = entity.get('morph_features') or entity.get('morph', {})
            
            # Znajdź token w tekście
            token = f"{{{entity_type}}}"
            token_pos = result_text.find(token)
            
            if token_pos == -1:
                continue
            
            # Pobierz odmienioną wartość
            new_value = self._get_value_for_entity(entity_type, morph_features, original_text)
            
            if new_value is None:
                continue
            
            # Podmień token
            result_text = (
                result_text[:token_pos] + 
                new_value + 
                result_text[token_pos + len(token):]
            )
        
        return result_text
    
    def generate_from_anon_text(self, anon_text: str) -> str:
        """
        Prostsza wersja - znajduje tokeny w tekście i zamienia je na losowe wartości.
        Używa mianownika (forma podstawowa).
        
        Args:
            anon_text: Tekst z tokenami typu {name}, {surname} itp.
        
        Returns:
            Tekst z podmienionymi tokenami
        """
        result_text = anon_text
        token_pattern = r'\{([^}]+)\}'
        
        def replace_token(match):
            entity_type = match.group(1).lower()
            new_value = self._get_value_for_entity(entity_type, {}, "")
            return new_value if new_value else match.group(0)
        
        result_text = re.sub(token_pattern, replace_token, result_text)
        return result_text
