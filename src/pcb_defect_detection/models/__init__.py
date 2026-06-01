"""Model definitions."""

from pcb_defect_detection.models.patch_cnn import PatchCNN
from pcb_defect_detection.models.single_image_cnn import SingleImageCNN

__all__ = ["PatchCNN", "SingleImageCNN"]
