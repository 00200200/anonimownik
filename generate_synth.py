import json
import random
from pathlib import Path

OUT = Path("data/processed/train_synth.json")
random.seed(42)


def add(text, spans):
    for s in spans:
        if text[s["start"] : s["end"]] != s["value"]:
            raise ValueError("Span mismatch")
    ents = [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in spans]
    return {"text": text, "entities": ents}


def place(text, value, label):
    i = text.index(value)
    return {"start": i, "end": i + len(value), "value": value, "label": label}


names = ["Michał", "Anna", "Jan", "Kasia", "Piotr", "Agnieszka"]
surnames = ["Nowak", "Kowalski", "Wiśniewski", "Zielińska", "Lewandowski"]
cities = ["Warszawa", "Kraków", "Gdańsk", "Wrocław", "Poznań"]
companies = ["CD Projekt", "Allegro", "InPost", "PKO BP", "Orlen"]
jobs = ["programista", "analityk", "kierownik", "tester", "architekt"]
schools = ["Politechnika Warszawska", "AGH", "UW", "UJ", "PG"]
politics = ["PiS", "Platforma Obywatelska", "Konfederacja", "Lewica"]
ethnic = ["Rom", "Żyd", "ukraińskiego pochodzenia"]
orient = ["gej", "lesbijka", "biseksualna", "heteroseksualny"]
relatives = ["mój brat", "moja siostra", "mój syn", "moja córka"]


def samples():
    data = []

    # IDs and contact
    for pesel in ["90010112345", "72031298765", "01234567890"]:
        text = f"Mój PESEL to {pesel}."
        data.append(add(text, [place(text, pesel, "pesel")]))
    for phone in ["501-234-567", "+48 600 700 800", "123 456 789"]:
        text = f"Zadzwoń na {phone}."
        data.append(add(text, [place(text, phone, "phone")]))
    for email in ["ala@example.com", "jan.kowalski@mail.pl", "user123@test.com"]:
        text = f"Napisz na {email}."
        data.append(add(text, [place(text, email, "email")]))
    for card in ["4111 1111 1111 1111", "5555-4444-3333-2222"]:
        text = f"Karta: {card}"
        data.append(add(text, [place(text, card, "credit-card-number")]))
    for bank in [
        "12 3456 7890 1234 5678 9012 3456",
        "45 1140 2004 0000 3202 1234 5678",
    ]:
        text = f"Numer konta {bank}"
        data.append(add(text, [place(text, bank, "bank-account")]))
    for doc in ["ABC123456", "FD1772", "XYZ987654"]:
        text = f"Dowód: {doc}"
        data.append(add(text, [place(text, doc, "document-number")]))

    # Dates
    for dob in ["1990-05-12", "1988/11/03", "2001-01-01"]:
        text = f"Data urodzenia {dob}"
        data.append(add(text, [place(text, dob, "date-of-birth")]))
    for dt in ["23.09.2023", "1 stycznia 2020", "14-07-2022"]:
        text = f"Wydarzenie było {dt}"
        data.append(add(text, [place(text, dt, "date")]))

    # Names, address, city
    for n in names:
        text = f"Nazywam się {n}."
        data.append(add(text, [place(text, n, "name")]))
    for s in surnames:
        text = f"Moje nazwisko to {s}."
        data.append(add(text, [place(text, s, "surname")]))
    for c in cities:
        text = f"Mieszkam w {c}."
        data.append(add(text, [place(text, c, "city")]))
    for addr in [
        "ul. Długa 5, 00-001 Warszawa",
        "ulica Prosta 12/3, 31-000 Kraków",
    ]:
        text = f"Adres: {addr}"
        data.append(add(text, [place(text, addr, "address")]))

    # Work/edu
    for comp in companies:
        text = f"Pracuję w {comp}."
        data.append(add(text, [place(text, comp, "company")]))
    for job in jobs:
        text = f"Moje stanowisko to {job}."
        data.append(add(text, [place(text, job, "job-title")]))
    for sch in schools:
        text = f"Uczyłem się na {sch}."
        data.append(add(text, [place(text, sch, "school-name")]))

    # Sensitive
    for p in politics:
        text = f"Popieram {p}."
        data.append(add(text, [place(text, p, "political-view")]))
    for e in ethnic:
        text = f"Jestem {e}."
        data.append(add(text, [place(text, e, "ethnicity")]))
    for o in orient:
        text = f"Jestem {o}."
        data.append(add(text, [place(text, o, "sexual-orientation")]))
    for r in relatives:
        text = f"{r} mieszka w Warszawie."
        data.append(add(text, [place(text, r, "relative")]))
    for sec in ["p$4mKg+vh5", "sekret123!", "Haslo!2024"]:
        text = f"Moje hasło to {sec}"
        data.append(add(text, [place(text, sec, "secret")]))
    for user in ["@janek123", "user_name42", "login-abc"]:
        text = f"Login: {user}"
        data.append(add(text, [place(text, user, "username")]))

    random.shuffle(data)
    return data


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = samples()
    with OUT.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(data)} samples to {OUT}")


if __name__ == "__main__":
    main()
