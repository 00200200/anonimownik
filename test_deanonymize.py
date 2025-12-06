#!/usr/bin/env python3
"""
Test skryptu deanonimizacji z pełnym użyciem modelu Herbert.
Model Herbert generuje: imiona, nazwiska, miasta, firmy, nazwy ulic, płeć.
"""

import sys
sys.path.append('/home/bartek/repos/animownik')

from src.scripts.deanonymize_herbert import HerbertDeanonymizer


def test_deanonymization():
    """Testuj deanonimizację na przykładowym tekście."""
    
    # Przykładowy tekst z tokenami
    sample_text = """
Sprawę prowadzi [name] [surname] Tel [phone] pok 005 
Otrzymują 1 Związek Międzygminny PUSZCZA ZIELONKA z załącznikami przez Pełnomocnika 
2 GMINA [city] [address] Do wiadomości 1 Burmistrz Miasta i Gminy w [city]
    
JEDNOSTKA PROJEKTUJĄCA [company] [address] tel [phone] fax [phone] 
NAZWA ZADANIA KANALIZACJA OBSZARU PARKU KRAJOBRAZOWEGO PUSZCZA ZIELONKA OKOLICE 
KONTRAKT IX Zadanie 6 POMPOWNIA MG_LOP_P610_007_PS1 BRANŻA TECHNOLOGIA 
PROJEKTOWAŁ mgr inż [name] [surname] SPRAWDZIŁ mgr inż [name] [surname] 
PRZEDMIOT RYSUNKU OPRACOWAŁ mgr inż [name] [surname] DATA [date]

„Cześć, muszę się komuś wygadać, bo już nie daję rady. Nazywam się [name] [surname], 
mam [age] lat ([sex]), mieszkam w [city] na [address]. Jakby co, to mój telefon to [phone], 
a mail [email] - ostatnio ciągle ktoś pyta o te dane, więc podaję od razu."
"""
    
    print("=== Test Deanonimizacji z Modelem Herbert ===\n")
    print("TEKST ORYGINALNY:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    # Inicjalizuj deanonimizator
    print("Ładuję model Herbert...")
    deanonymizer = HerbertDeanonymizer()
    
    print("\nPrzetwarzam tekst...\n")
    result = deanonymizer.deanonymize_text(sample_text)
    
    print("TEKST DEANONIMIZOWANY:")
    print(result)
    print("\n" + "="*50)
    print("\nTest zakończony!")


if __name__ == '__main__':
    test_deanonymization()
