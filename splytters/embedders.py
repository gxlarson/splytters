from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np


def _features_to_numpy(outputs: Any) -> np.ndarray:
    """Convert a CLIP feature output to a numpy array.

    transformers < 5 returns a tensor from ``get_*_features``; transformers >= 5
    returns a ``BaseModelOutputWithPooling`` — handle both.
    """
    if hasattr(outputs, "detach"):
        tensor = outputs
    elif getattr(outputs, "pooler_output", None) is not None:
        tensor = outputs.pooler_output
    else:
        tensor = outputs.last_hidden_state.mean(dim=1)
    return tensor.detach().cpu().numpy()


class Embedder(ABC):
    """Base class for all embedders."""

    @abstractmethod
    def embed(self, inputs: Sequence[Any]) -> np.ndarray:
        """Embed a list of inputs. Returns np.ndarray of shape (n, dim)."""


def list_embedders() -> list[str]:
    """Return the names of the available concrete embedder classes.

    These live in :mod:`splytters.embedders` (the ``[embedders]`` extra). The
    underlying model libraries (sentence-transformers, transformers, openai)
    are imported lazily only when an embedder is constructed, so listing pulls
    in no optional dependency.

    Returns:
        Embedder class names, e.g. ``["TextEmbedder", "CLIPTextEmbedder",
        "CLIPImageEmbedder", "OpenAIEmbedder"]``.
    """
    return [cls.__name__ for cls in Embedder.__subclasses__()]


class TextEmbedder(Embedder):
    """Embed text using a SentenceTransformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)


class CLIPTextEmbedder(Embedder):
    """Embed text using a CLIP model."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        from transformers import CLIPModel, CLIPTokenizerFast

        self.model = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model.get_text_features(**inputs)
        return _features_to_numpy(outputs)


class CLIPImageEmbedder(Embedder):
    """Embed images using a CLIP model."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed(self, images: Sequence[Any]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        return _features_to_numpy(outputs)


class OpenAIEmbedder(Embedder):
    """Embed text using the OpenAI embeddings API."""

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = model_name

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        # The API does not guarantee response order; each item's ``index`` gives
        # its position in ``texts``, so sort by it before stacking to keep the
        # returned rows aligned with the input.
        data = sorted(response.data, key=lambda item: item.index)
        return np.array([item.embedding for item in data])
