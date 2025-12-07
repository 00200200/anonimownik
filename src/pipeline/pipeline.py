import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .anonymizer import Anonymizer, AnonymizationResult, EntityMeta
from .synthetic_generator import SyntheticGenerator


@dataclass
class PipelineResult:

    original_text: str
    anonymized_text: str
    synthetic_text: str
    entities: List[EntityMeta] = field(default_factory=list)

    entity_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "original_text": self.original_text,
            "anonymized_text": self.anonymized_text,
            "synthetic_text": self.synthetic_text,
            "entities": [e.to_dict() for e in self.entities],
            "entity_counts": self.entity_counts,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class BatchResult:
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

    def __init__(
        self,
        ner_model_path: Optional[str] = None,
        spacy_model: str = "pl_core_news_sm",
        device: Optional[str] = None,
        seed: Optional[int] = None,
        use_regex_rules: bool = True,
    ):
        self.use_regex_rules = use_regex_rules

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

    def _is_already_anonymized(self, text: str) -> bool:
        """
        Sprawdza, czy tekst jest już anonimizowany (zawiera tokeny w nawiasach).

        Args:
            text: Tekst do sprawdzenia

        Returns:
            True jeśli tekst zawiera tokeny w nawiasach kwadratowych lub klamrowych
        """
        # Sprawdź czy są tokeny w nawiasach kwadratowych [name] lub klamrowych {name}
        square_bracket_pattern = r"\[([^\]]+)\]"
        curly_bracket_pattern = r"\{([^}]+)\}"

        has_square = bool(re.search(square_bracket_pattern, text))
        has_curly = bool(re.search(curly_bracket_pattern, text))

        return has_square or has_curly

    def process(self, text: str) -> PipelineResult:
        """
        Przetwarza pojedynczy tekst przez cały pipeline.

        Jeśli tekst jest już anonimizowany (zawiera tokeny w nawiasach),
        pomija etap anonimizacji i przechodzi od razu do generacji syntetycznej.

        Args:
            text: Tekst do przetworzenia

        Returns:
            PipelineResult z wynikami wszystkich etapów
        """
        # Sprawdź czy tekst jest już anonimizowany
        if self._is_already_anonymized(text):
            # Tekst jest już anonimizowany - tylko generuj syntetyczny
            anon_text = text
            synthetic_text = self.generator.generate_from_anon_text(text)

            # Wyciągnij encje na podstawie istniejących tokenów w nawiasach kwadratowych
            entities_meta: List[EntityMeta] = []
            for m in re.finditer(r"\[([^\]]+)\]", text):
                label = m.group(1)
                entities_meta.append(
                    EntityMeta(
                        start=m.start(),
                        end=m.end(),
                        entity_type=label,
                        original_text="",
                        morph_features={},
                    )
                )

            entity_counts: Dict[str, int] = {}
            for ent in entities_meta:
                entity_counts[ent.entity_type] = (
                    entity_counts.get(ent.entity_type, 0) + 1
                )

            return PipelineResult(
                original_text=text,
                anonymized_text=anon_text,
                synthetic_text=synthetic_text,
                entities=entities_meta,
                entity_counts=entity_counts,
            )

        # Normalny proces: anonimizacja + generacja syntetyczna
        # Etap 1: Anonimizacja
        anon_result = self.anonymizer.anonymize(text, use_regex=self.use_regex_rules)

        # Etap 2: Generacja syntetyczna
        synthetic_text = self.generator.generate(
            anon_result.anon_text, anon_result.entities_meta
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
        results = []
        total_entities = 0
        entity_type_counts: Dict[str, int] = {}

        for text in texts:
            result = self.process(text)
            results.append(result)

            total_entities += len(result.entities)
            for entity_type, count in result.entity_counts.items():
                entity_type_counts[entity_type] = (
                    entity_type_counts.get(entity_type, 0) + count
                )

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
        input_path = Path(input_path)

        # Wczytaj tekst
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Podziel na części
        if split_paragraphs:
            texts = [p.strip() for p in content.split("\n\n") if p.strip()]
        else:
            texts = [content]

        # Przetwórz
        result = self.process_batch(texts)

        # Zapisz wynik
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_format in ("json", "both"):
                json_path = (
                    output_path.with_suffix(".json")
                    if output_format == "both"
                    else output_path
                )
                with open(json_path, "w", encoding="utf-8") as f:
                    f.write(result.to_json())

            if output_format in ("text", "both"):
                text_path = (
                    output_path.with_suffix(".txt")
                    if output_format == "both"
                    else output_path
                )
                with open(text_path, "w", encoding="utf-8") as f:
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
        return self.anonymizer.anonymize(text, use_regex=self.use_regex_rules)

    def generate_synthetic(
        self,
        anon_text: str,
        entities_meta: Optional[List[EntityMeta]] = None,
    ) -> str:
        if entities_meta:
            return self.generator.generate(anon_text, entities_meta)
        else:
            return self.generator.generate_from_anon_text(anon_text)


def create_pipeline(
    model_path: Optional[str] = None, seed: Optional[int] = None, **kwargs
) -> AnonymizationPipeline:
    # Sprawdź domyślną ścieżkę modelu
    if model_path is None:
        default_model = Path("models/herbert-ner")
        if default_model.exists():
            model_path = str(default_model)

    return AnonymizationPipeline(ner_model_path=model_path, seed=seed, **kwargs)
