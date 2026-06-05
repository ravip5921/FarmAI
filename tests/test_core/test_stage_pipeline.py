from __future__ import annotations

import unittest

from src.core.image import DocumentImage
from src.core.pipeline import Pipeline
from src.core.stage import PipelineStage


class IncrementStage(PipelineStage):
    def __init__(self, amount: int = 1):
        self.amount = amount

    def process(self, doc: DocumentImage) -> DocumentImage:
        # assume image is an int for these tests
        new_image = doc.image + self.amount
        metadata = dict(doc.metadata)
        order = metadata.get("order", [])
        order = list(order) + [f"inc_{self.amount}"]
        metadata["order"] = order
        return DocumentImage(image=new_image, metadata=metadata)


class TestPipelineAndStage(unittest.TestCase):
    def test_pipeline_add_and_run(self) -> None:
        p = Pipeline()
        p.add_stage(IncrementStage(2))
        p.add_stage(IncrementStage(3))

        doc = DocumentImage(image=0, metadata={})
        out = p.run(doc)

        self.assertEqual(out.image, 5)
        self.assertEqual(out.metadata.get("order"), ["inc_2", "inc_3"])

    def test_pipeline_extend_and_call(self) -> None:
        p = Pipeline()
        stages = [IncrementStage(1), IncrementStage(4)]
        p.extend(stages)

        doc = DocumentImage(image=10, metadata={"order": []})
        out = p(doc)  # __call__ delegates to run

        self.assertEqual(out.image, 15)
        self.assertEqual(out.metadata.get("order"), ["inc_1", "inc_4"])

    def test_pipelinestage_is_abstract(self) -> None:
        # PipelineStage should be abstract and not instantiable
        with self.assertRaises(TypeError):
            PipelineStage()

    def test_pipeline_stage_super_raises_not_implemented(self) -> None:
        # Create a concrete subclass that calls the base implementation
        class BrokenStage(PipelineStage):
            def process(self, doc: DocumentImage) -> DocumentImage:
                return super().process(doc)

        broken = BrokenStage()
        with self.assertRaises(NotImplementedError):
            broken.process(DocumentImage(image=0, metadata={}))


if __name__ == "__main__":
    unittest.main()
