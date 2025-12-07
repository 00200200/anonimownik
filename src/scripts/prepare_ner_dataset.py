import json
import re
from pathlib import Path
from typing import List, Dict, Tuple


def extract_entities_from_pair(original: str, anonymized: str) -> List[Dict]:
    entities = []
    token_pattern = r"\[([^\]]+)\]"
    tokens = list(re.finditer(token_pattern, anonymized))

    if not tokens:
        return entities

    orig_idx = 0
    anon_idx = 0

    for token_match in tokens:
        token_start = token_match.start()
        token_end = token_match.end()
        label = token_match.group(1)

        before_token = anonymized[anon_idx:token_start]

        if before_token:
            found_pos = original.find(before_token, orig_idx)
            if found_pos == -1:
                continue
            orig_idx = found_pos + len(before_token)

        entity_start = orig_idx
        anon_idx = token_end

        next_token = re.search(token_pattern, anonymized[anon_idx:])
        if next_token:
            next_static = anonymized[anon_idx : anon_idx + next_token.start()]
        else:
            next_static = anonymized[anon_idx:]

        if next_static:
            next_pos = original.find(next_static, orig_idx)
            if next_pos != -1:
                entity_end = next_pos
                entity_text = original[entity_start:entity_end]
            else:
                words = original[orig_idx:].split(None, 2)
                if words:
                    entity_text = words[0]
                    entity_end = orig_idx + len(words[0])
                else:
                    continue
        else:
            entity_text = original[orig_idx:].strip()
            entity_end = len(original)

        if entity_text.strip():
            entities.append({"start": entity_start, "end": entity_end, "label": label})

            orig_idx = entity_end

    return entities


def load_and_process_data(orig_path: str, anon_path: str) -> List[Dict]:
    dataset = []

    with (
        open(orig_path, "r", encoding="utf-8") as f_orig,
        open(anon_path, "r", encoding="utf-8") as f_anon,
    ):

        for line_num, (orig_line, anon_line) in enumerate(zip(f_orig, f_anon), 1):
            orig_line = orig_line.strip()
            anon_line = anon_line.strip()

            if not orig_line or not anon_line:
                continue

            entities = extract_entities_from_pair(orig_line, anon_line)
            dataset.append(
                {"text": orig_line, "entities": entities, "line_number": line_num}
            )

            if line_num % 100 == 0:
                print(f"Przetworzono {line_num} linii...")

    return dataset


def split_train_test(
    dataset: List[Dict], test_ratio: float = 0.2
) -> Tuple[List[Dict], List[Dict]]:
    import random

    random.seed(42)

    shuffled = dataset.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - test_ratio))
    train_data = shuffled[:split_idx]
    test_data = shuffled[split_idx:]

    return train_data, test_data


def save_to_json(data: List[Dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Zapisano {len(data)} przykładów do {output_path}")


def main():
    orig_path = "data/nask_train/anonymized.txt"
    anon_path = "data/nask_train/orig.txt"
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Przetwarzanie danych...")
    dataset = load_and_process_data(orig_path, anon_path)
    print(f"Przetworzono {len(dataset)} linii")

    train_data, test_data = split_train_test(dataset, test_ratio=0.05)

    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"

    save_to_json(train_data, str(train_path))
    save_to_json(test_data, str(test_path))


if __name__ == "__main__":
    main()
