#!/usr/bin/env python3
"""
Skrypt demonstracyjny pokazujący różnicę między starą a nową logiką treningu.
"""

print("=" * 70)
print("DEMONSTRACJA: Stara vs Nowa logika treningu")
print("=" * 70)

# Przykładowy tekst z tokenami
example_text = "Nazywam się [name] [surname], mieszkam w [city] i mój PESEL to [pesel]."

print("\n📋 TEKST ORYGINALNY:")
print(f"   {example_text}")

print("\n" + "=" * 70)
print("STARA LOGIKA (bez uwzględnienia typu tokenu)")
print("=" * 70)

old_masked = example_text.replace("[name]", "[MASK]")
old_masked = old_masked.replace("[surname]", "[MASK]")
old_masked = old_masked.replace("[city]", "[MASK]")
old_masked = old_masked.replace("[pesel]", "[MASK]")

print("\n🔴 Tekst dla modelu (STARY):")
print(f"   {old_masked}")
print("\n❌ Problem:")
print("   Model widzi tylko [MASK] [MASK], mieszkam w [MASK] i mój PESEL to [MASK]")
print("   Nie wie, które [MASK] to imię, a które PESEL!")
print("   Wszystkie tokeny wyglądają identycznie.")

print("\n" + "=" * 70)
print("NOWA LOGIKA (z uwzględnieniem typu tokenu)")
print("=" * 70)

new_masked = example_text.replace("[name]", "[MASK:name]")
new_masked = new_masked.replace("[surname]", "[MASK:surname]")
new_masked = new_masked.replace("[city]", "[MASK:city]")
new_masked = new_masked.replace("[pesel]", "[MASK:pesel]")

print("\n🟢 Tekst dla modelu (NOWY):")
print(f"   {new_masked}")
print("\n✅ Zalety:")
print("   1. Model wie, że [MASK:name] to imię")
print("   2. Model wie, że [MASK:surname] to nazwisko")
print("   3. Model wie, że [MASK:city] to miasto")
print("   4. Model wie, że [MASK:pesel] to numer PESEL")
print("\n   Każdy typ ma swój unikalny token - model może się nauczyć")
print("   kontekstu charakterystycznego dla każdego typu danych!")

print("\n" + "=" * 70)
print("WPŁYW NA JAKOŚĆ PREDYKCJI")
print("=" * 70)

print("\n🎯 STARA METODA:")
print("   Kontekst: 'Nazywam się [MASK]'")
print("   Model myśli: To może być imię? nazwisko? miasto? PESEL?")
print("   Trudność: ★★★★★")

print("\n🎯 NOWA METODA:")
print("   Kontekst: 'Nazywam się [MASK:name]'")
print("   Model wie: To musi być IMIĘ! Szukam polskiego imienia w kontekście.")
print("   Trudność: ★★☆☆☆")

print("\n" + "=" * 70)
print("PRZYKŁAD Z WIELOMA TOKENAMI")
print("=" * 70)

complex_text = "Mieszkam w [city] przy [address], tel: [phone], mail: [email]"

print(f"\n📋 Tekst: {complex_text}")

print("\n🔴 STARE (wszystkie [MASK]):")
old_complex = complex_text.replace("[city]", "[MASK]").replace("[address]", "[MASK]")
old_complex = old_complex.replace("[phone]", "[MASK]").replace("[email]", "[MASK]")
print(f"   {old_complex}")
print("   ❌ Model zgubi się - wszystko wygląda tak samo!")

print("\n🟢 NOWE (z typami):")
new_complex = complex_text.replace("[city]", "[MASK:city]")
new_complex = new_complex.replace("[address]", "[MASK:address]")
new_complex = new_complex.replace("[phone]", "[MASK:phone]")
new_complex = new_complex.replace("[email]", "[MASK:email]")
print(f"   {new_complex}")
print("   ✅ Model wie dokładnie, czego szukać w każdym miejscu!")

print("\n" + "=" * 70)
print("WNIOSKI")
print("=" * 70)
print("""
✨ Nowa metoda daje modelowi więcej informacji:
   • Model uczy się charakterystycznych kontekstów dla każdego typu
   • Nie musi "zgadywać" typu na podstawie tylko kontekstu
   • Może lepiej wykorzystać embeddingi specyficzne dla typu
   • Wyższa dokładność predykcji

🚀 Dzięki temu model Herbert może:
   • Generować bardziej trafne imiona w kontekście imion
   • Generować bardziej realistyczne miasta w kontekście miast
   • Nie mylić numerów PESEL z numerami telefonów
   • Lepiej dopasować format danych do typu (email ma @, telefon ma cyfry)
""")

print("=" * 70)
