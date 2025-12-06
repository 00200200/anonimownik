import json
from typing import List, Dict, Tuple
from pathlib import Path


def load_ner_dataset(filepath: str) -> List[Dict]:
    dataset = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def load_train_test() -> Tuple[List[Dict], List[Dict]]:
    train_data = load_ner_dataset("data/processed/train.json")
    test_data = load_ner_dataset("data/processed/test.json")
    return train_data, test_data


def get_entity_text(text: str, entity: Dict) -> str:
    return text[entity["start"] : entity["end"]]


def load_first_n_lines(n=30):
    with open("data/nask_train/orig.txt", "r", encoding="utf-8") as f:
        orig_lines = [f.readline().strip() for _ in range(n)]

    with open("data/nask_train/anonymized.txt", "r", encoding="utf-8") as f:
        anon_lines = [f.readline().strip() for _ in range(n)]

    return orig_lines, anon_lines


if __name__ == "__main__":
    train_data, test_data = load_train_test()
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    example = next((ex for ex in train_data if ex["entities"]), None)
    if example:
        print(f"\nText: {example['text'][:100]}...")
        for ent in example["entities"][:3]:
            text = get_entity_text(example["text"], ent)
            print(f"  {ent['label']}: {text}")
