from importlib.metadata import metadata

from .batch import Request, create_batch, get_inlined_responses

_package_metadata = metadata(str(__package__))  # ruff:ignore[non-empty-init-module]
__version__ = _package_metadata["Version"]
__summary__ = _package_metadata["Summary"]
__author__ = _package_metadata.get("Author-email", "")

__all__ = [
    "Request",
    "__author__",
    "__summary__",
    "__version__",
    "create_batch",
    "get_inlined_responses",
]
