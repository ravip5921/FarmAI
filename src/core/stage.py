from __future__ import annotations

from abc import ABC, abstractmethod

from .image import DocumentImage


class PipelineStage(ABC):
	@abstractmethod
	def process(self, doc: DocumentImage) -> DocumentImage:
		raise NotImplementedError
