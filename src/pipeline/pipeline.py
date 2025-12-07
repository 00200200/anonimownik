"""Główny pipeline anonimizacji i generacji syntetycznej.

Łączy moduł anonimizacji (NER + reguły) z generatorem syntetycznym
w jeden spójny workflow.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .anonymizer import Anonymizer, AnonymizationResult, EntityMeta
from .synthetic_generator import SyntheticGenerator


@dataclass
class PipelineResult:
    """Wynik przetworzenia pipeline'u."""
    
    # Tekst wejściowy
    original_text: str
    
    # Tekst zanonimizowany (z tokenami typu {name})
    anonymized_text: str
    
    # Tekst syntetyczny (z nowymi danymi)
    synthetic_text: str
    
    # Metadane wykrytych encji
    entities: List[EntityMeta] = field(default_factory=list)
    
    # Statystyki
    entity_counts: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Konwertuje wynik do słownika."""
        return {
            "original_text": self.original_text,
            "anonymized_text": self.anonymized_text,
            "synthetic_text": self.synthetic_text,
            "entities": [e.to_dict() for e in self.entities],
            "entity_counts": self.entity_counts,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Konwertuje wynik do JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class BatchResult:
    """Wynik przetworzenia wielu tekstów."""
    
    results: List[PipelineResult] = field(default_factory=list)
    total_entities: int = 0
    entity_type_counts: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_entities": self.total_entities,
            "entity_type_counts": self.entity_type_counts,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class AnonymizationPipeline:
    """
    Pipeline anonimizacji i generacji syntetycznej dla języka polskiego.
    
    Etapy:
    1. Anonimizacja (NER + reguły) - wykrycie encji i zastąpienie tokenami
    2. Generacja syntetyczna - podmiana tokenów na nowe, odmienione wartości
    
    Przykład użycia:
        >>> pipeline = AnonymizationPipeline()
        >>> result = pipeline.process("Jestem Jan Kowalski z Warszawy.")
        >>> print(result.anonymized_text)  # "Jestem {name} {surname} z {city}."
        >>> print(result.synthetic_text)   # "Jestem Piotr Nowak z Krakowa."
    """
    
    def __init__(
        self,
        ner_model_path: Optional[str] = None,
        spacy_model: str = "pl_core_news_sm",
        device: Optional[str] = None,
        seed: Optional[int] = None,
        use_regex_rules: bool = True,
    ):
        """
        Inicjalizuje pipeline.
        
        Args:
            ner_model_path: Ścieżka do wytrenowanego modelu NER (opcjonalne)
            spacy_model: Nazwa modelu spaCy do analizy morfologicznej
            device: Urządzenie (cuda/cpu), automatyczne jeśli None
            seed: Ziarno losowości dla generatora syntetycznego
            use_regex_rules: Czy używać reguł regex do wykrywania encji
        """
        self.use_regex_rules = use_regex_rules
        
        # Inicjalizuj anonimizer
        self.anonymizer = Anonymizer(
            model_path=ner_model_path,
            spacy_model=spacy_model,
            device=device,
        )
        
        # Inicjalizuj generator syntetyczny
        self.generator = SyntheticGenerator(
            seed=seed,
            spacy_model=spacy_model,
        )
    
    def process(self, text: str) -> PipelineResult:
        """
        Przetwarza pojedynczy tekst przez cały pipeline.
        
        Args:
            text: Tekst do przetworzenia
            
        Returns:
            PipelineResult z wynikami wszystkich etapów
        """
        # Etap 1: Anonimizacja
        anon_result = self.anonymizer.anonymize(text, use_regex=self.use_regex_rules)
        
        # Etap 2: Generacja syntetyczna
        synthetic_text = self.generator.generate(
            anon_result.anon_text,
            anon_result.entities_meta
        )
        
        # Oblicz statystyki
        entity_counts: Dict[str, int] = {}
        for entity in anon_result.entities_meta:
            entity_type = entity.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        return PipelineResult(
            original_text=text,
            anonymized_text=anon_result.anon_text,
            synthetic_text=synthetic_text,
            entities=anon_result.entities_meta,
            entity_counts=entity_counts,
        )
    
    def process_batch(self, texts: List[str]) -> BatchResult:
        """
        Przetwarza wiele tekstów.
        
        Args:
            texts: Lista tekstów do przetworzenia
            
        Returns:
            BatchResult z wynikami dla wszystkich tekstów
        """
        results = []
        total_entities = 0
        entity_type_counts: Dict[str, int] = {}
        
        for text in texts:
            result = self.process(text)
            results.append(result)
            
            total_entities += len(result.entities)
            for entity_type, count in result.entity_counts.items():
                entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + count
        
        return BatchResult(
            results=results,
            total_entities=total_entities,
            entity_type_counts=entity_type_counts,
        )
    
    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        output_format: str = "json",
        split_paragraphs: bool = True,
    ) -> BatchResult:
        """
        Przetwarza plik tekstowy.
        
        Args:
            input_path: Ścieżka do pliku wejściowego
            output_path: Ścieżka do pliku wyjściowego (opcjonalna)
            output_format: Format wyjścia ('json', 'text', 'both')
            split_paragraphs: Czy dzielić na akapity
            
        Returns:
            BatchResult z wynikami
        """
        input_path = Path(input_path)
        
        # Wczytaj tekst
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Podziel na części
        if split_paragraphs:
            texts = [p.strip() for p in content.split('\n\n') if p.strip()]
        else:
            texts = [content]
        
        # Przetwórz
        result = self.process_batch(texts)
        
        # Zapisz wynik
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_format in ('json', 'both'):
                json_path = output_path.with_suffix('.json') if output_format == 'both' else output_path
                with open(json_path, 'w', encoding='utf-8') as f:
                    f.write(result.to_json())
            
            if output_format in ('text', 'both'):
                text_path = output_path.with_suffix('.txt') if output_format == 'both' else output_path
                with open(text_path, 'w', encoding='utf-8') as f:
                    for r in result.results:
                        f.write("=== ORYGINALNY ===\n")
                        f.write(r.original_text + "\n\n")
                        f.write("=== ANONIMIZOWANY ===\n")
                        f.write(r.anonymized_text + "\n\n")
                        f.write("=== SYNTETYCZNY ===\n")
                        f.write(r.synthetic_text + "\n\n")
                        f.write("-" * 50 + "\n\n")
        
        return result
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        pattern: str = "*.txt",
        output_format: str = "json",
    ) -> Dict[str, BatchResult]:
        """
        Przetwarza wszystkie pliki w katalogu.
        
        Args:
            input_dir: Katalog wejściowy
            output_dir: Katalog wyjściowy (opcjonalny)
            pattern: Wzorzec nazw plików (np. "*.txt")
            output_format: Format wyjścia
            
        Returns:
            Słownik {nazwa_pliku: BatchResult}
        """
        input_dir = Path(input_dir)
        results = {}
        
        for file_path in input_dir.glob(pattern):
            if file_path.is_file():
                output_path = None
                if output_dir:
                    output_dir = Path(output_dir)
                    output_path = output_dir / file_path.name
                
                results[file_path.name] = self.process_file(
                    file_path, 
                    output_path,
                    output_format=output_format,
                )
        
        return results
    
    def anonymize_only(self, text: str) -> AnonymizationResult:
        """
        Wykonuje tylko anonimizację (bez generacji syntetycznej).
        
        Args:
            text: Tekst do anonimizacji
            
        Returns:
            AnonymizationResult z anon_text i entities_meta
        """
        return self.anonymizer.anonymize(text, use_regex=self.use_regex_rules)
    
    def generate_synthetic(
        self,
        anon_text: str,
        entities_meta: Optional[List[EntityMeta]] = None,
    ) -> str:
        """
        Generuje tekst syntetyczny z zanonimizowanego tekstu.
        
        Args:
            anon_text: Tekst z tokenami ({name}, {surname} itp.)
            entities_meta: Metadane encji (opcjonalne, dla odmiany)
            
        Returns:
            Tekst z podmienionymi tokenami
        """
        if entities_meta:
            return self.generator.generate(anon_text, entities_meta)
        else:
            return self.generator.generate_from_anon_text(anon_text)


def create_pipeline(
    model_path: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs
) -> AnonymizationPipeline:
    """
    Fabryka do tworzenia pipeline'u z domyślnymi ustawieniami.
    
    Args:
        model_path: Ścieżka do modelu NER
        seed: Ziarno losowości
        **kwargs: Dodatkowe argumenty dla AnonymizationPipeline
        
    Returns:
        Skonfigurowany AnonymizationPipeline
    """
    # Sprawdź domyślną ścieżkę modelu
    if model_path is None:
        default_model = Path("models/herbert-ner")
        if default_model.exists():
            model_path = str(default_model)
    
    return AnonymizationPipeline(
        ner_model_path=model_path,
        seed=seed,
        **kwargs
    )

