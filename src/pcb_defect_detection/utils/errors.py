"""Custom exceptions with user-friendly messages."""


class PCBDefectDetectionError(Exception):
    """Base class for expected project errors."""


class ImageLoadError(PCBDefectDetectionError):
    """Raised when an image cannot be read or decoded."""


class InputValidationError(PCBDefectDetectionError):
    """Raised when input files or image pairs are invalid."""


class DatasetError(PCBDefectDetectionError):
    """Raised when a dataset or manifest is malformed."""


class ModelLoadError(PCBDefectDetectionError):
    """Raised when a model checkpoint is missing or incompatible."""
