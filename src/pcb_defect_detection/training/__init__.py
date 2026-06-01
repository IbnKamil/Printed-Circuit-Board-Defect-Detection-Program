"""Training and evaluation pipelines."""

from pcb_defect_detection.training.evaluate_detector import evaluate_detector
from pcb_defect_detection.training.train_patch_classifier import train_patch_classifier
from pcb_defect_detection.training.train_single_image_classifier import train_single_image_classifier

__all__ = ["evaluate_detector", "train_patch_classifier", "train_single_image_classifier"]
