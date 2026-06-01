"""PyTorch datasets built from DeepPCB pair annotations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from pcb_defect_detection.data.deeppcb import parse_annotation_file
from pcb_defect_detection.data.io import load_image
from pcb_defect_detection.models.patch_cnn import make_patch_tensor
from pcb_defect_detection.preprocessing.transforms import PairPreprocessor
from pcb_defect_detection.utils.errors import DatasetError
from pcb_defect_detection.utils.schemas import ImagePair


@dataclass(frozen=True)
class PatchSample:
    """One supervised defect patch sample."""

    pair_index: int
    annotation_index: int


class DeepPCBPatchDataset(Dataset):
    """Dataset of cropped template/tested patches around annotated defects."""

    def __init__(
        self,
        pairs: list[ImagePair],
        image_size: int = 64,
        margin: int = 8,
    ) -> None:
        self.pairs = [pair for pair in pairs if pair.annotation_path is not None]
        if not self.pairs:
            raise DatasetError("No annotated image pairs are available for training.")
        self.image_size = image_size
        self.margin = margin
        self.preprocessor = PairPreprocessor()
        self.annotations = [
            parse_annotation_file(pair.annotation_path) for pair in self.pairs
        ]
        self.samples = [
            PatchSample(pair_index=pair_index, annotation_index=index)
            for pair_index, _pair in enumerate(self.pairs)
            for index, _annotation in enumerate(self.annotations[pair_index])
        ]
        if not self.samples:
            raise DatasetError("No annotations were found in provided image pairs.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        pair = self.pairs[sample.pair_index]
        annotation = self.annotations[sample.pair_index][sample.annotation_index]
        template = load_image(pair.template_path)
        tested = load_image(pair.test_path)
        preprocessed = self.preprocessor(template, tested)
        tensor = make_patch_tensor(
            preprocessed.template_gray,
            preprocessed.tested_gray,
            annotation.bbox,
            image_size=self.image_size,
            margin=self.margin,
        )
        return tensor, torch.tensor(annotation.class_id, dtype=torch.long)
