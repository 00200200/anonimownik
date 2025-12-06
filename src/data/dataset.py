"""Dataset do treningu modelu MLM na danych deanonimizacji."""

from typing import List
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class DeanonymizationDataset(Dataset):
    """Dataset zawierający pary: tekst oryginalny (z tokenami) -> tekst deanonimizowany."""
    
    def __init__(
        self, 
        orig_texts: List[str], 
        anon_texts: List[str], 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256
    ):
        """
        Args:
            orig_texts: Lista tekstów z tokenami [name], [surname] itp.
            anon_texts: Lista tekstów wypełnionych (deanonimizowanych)
            tokenizer: Tokenizer HuggingFace
            max_length: Maksymalna długość sekwencji
        """
        assert len(orig_texts) == len(anon_texts), "orig_texts i anon_texts muszą mieć tę samą długość"
        
        self.orig_texts = orig_texts
        self.anon_texts = anon_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.anon_texts)
    
    def __getitem__(self, idx: int):
        """Zwraca ztokenizowany tekst deanonimizowany (do MLM)."""
        text = self.anon_texts[idx]
        
        # Tokenizuj
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Usuń wymiar batch (zwracamy pojedynczy przykład)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def load_data(orig_path: str, anonymized_path: str) -> tuple[List[str], List[str]]:
    """
    Wczytaj dane z plików.
    
    Args:
        orig_path: Ścieżka do pliku z tokenami
        anonymized_path: Ścieżka do pliku z danymi deanonimizowanymi
        
    Returns:
        Tuple (orig_texts, anon_texts) - listy akapitów
    """
    print(f"Ładowanie danych z {orig_path} i {anonymized_path}...")
    
    with open(orig_path, 'r', encoding='utf-8') as f:
        orig_text = f.read()
    
    with open(anonymized_path, 'r', encoding='utf-8') as f:
        anon_text = f.read()
    
    # Podziel na akapity
    orig_paragraphs = [p.strip() for p in orig_text.split('\n') if p.strip()]
    anon_paragraphs = [p.strip() for p in anon_text.split('\n') if p.strip()]
    
    print(f"Załadowano {len(orig_paragraphs)} akapitów z orig.txt")
    print(f"Załadowano {len(anon_paragraphs)} akapitów z anonymized.txt")
    
    # Upewnij się że mamy tyle samo
    min_len = min(len(orig_paragraphs), len(anon_paragraphs))
    return orig_paragraphs[:min_len], anon_paragraphs[:min_len]
