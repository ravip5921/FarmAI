from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .image import DocumentImage
from .stage import PipelineStage


@dataclass
class Pipeline:
    stages: list[PipelineStage] = field(default_factory=list)

    def add_stage(self, stage: PipelineStage) -> None:
        self.stages.append(stage)

    def extend(self, stages: Iterable[PipelineStage]) -> None:
        self.stages.extend(stages)

    def run(self, doc: DocumentImage) -> DocumentImage:
        for stage in self.stages:
            doc = stage.process(doc)
        return doc

    def __call__(self, doc: DocumentImage) -> DocumentImage:
        return self.run(doc)
