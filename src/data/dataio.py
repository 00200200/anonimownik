def load_first_n_lines(n=30):
    with open("data/nask_train/orig.txt", "r", encoding="utf-8") as f:
        orig_lines = [f.readline().strip() for _ in range(n)]

    with open("data/nask_train/anonymized.txt", "r", encoding="utf-8") as f:
        anon_lines = [f.readline().strip() for _ in range(n)]

    return orig_lines, anon_lines


if __name__ == "__main__":
    original, anonymized = load_first_n_lines(30)

    for i, (orig, anon) in enumerate(zip(original, anonymized), 1):
        print(f"ORIG: {orig[:150]}..." if len(orig) > 150 else f"ORIG: {orig}")
        print(f"ANON: {anon[:150]}..." if len(anon) > 150 else f"ANON: {anon}")
