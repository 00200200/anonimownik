from src.model.model_encrypter import NERModel
from src.scripts.train_bert import ID2LABEL


def load_model(checkpoint_path: str = "models/ner_model.ckpt"):
    model = NERModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def test_model():
    model = load_model()

    test_texts = [
        "Nazywam się Jan Kowalski, mam 30 lat.",
        "Mój PESEL to 90010112345, tel 501222333",
        "Mieszkam w Warszawie przy ul. Długiej 5.",
    ]

    for text in test_texts:
        print(f"\nTekst: {text}")
        entities = model.predict_entities(text, ID2LABEL)

        if entities:
            for ent in entities:
                value = text[ent["start"] : ent["end"]]
                print(f"  {ent['label']}: {value}")


if __name__ == "__main__":
    test_model()
