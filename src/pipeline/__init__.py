"""Pipeline anonimizacji i generacji syntetycznej dla języka polskiego.

Główne komponenty:
- Anonymizer: Wykrywanie i anonimizacja encji (NER + reguły)
- SyntheticGenerator: Generacja syntetycznych danych z odmianą morfologiczną
- AnonymizationPipeline: Pełny pipeline łączący oba moduły

Przykład użycia:
    >>> from pipeline import AnonymizationPipeline
    >>> pipe = AnonymizationPipeline()
    >>> result = pipe.process("Jestem Jan Kowalski z Warszawy.")
    >>> print(result.anonymized_text)  # "Jestem {name} {surname} z {city}."
    >>> print(result.synthetic_text)   # "Jestem Piotr Nowak z Krakowa."
    
CLI:
    python -m pipeline --text "Jestem Anna Nowak."
    python -m pipeline --input plik.txt --output wynik.json
"""

from .anonymizer import Anonymizer, AnonymizationResult, EntityMeta
from .synthetic_generator import SyntheticGenerator
from .pipeline import AnonymizationPipeline, PipelineResult, BatchResult, create_pipeline

__all__ = [
    # Anonimizacja
    "Anonymizer",
    "AnonymizationResult",
    "EntityMeta",
    # Generacja syntetyczna
    "SyntheticGenerator",
    # Pipeline
    "AnonymizationPipeline",
    "PipelineResult",
    "BatchResult",
    "create_pipeline",
]
