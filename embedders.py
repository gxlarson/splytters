from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Base class for all embedders."""

    @abstractmethod
    def embed(self, inputs) -> np.ndarray:
        """Embed a list of inputs. Returns np.ndarray of shape (n, dim)."""


class TextEmbedder(Embedder):
    """Embed text using a SentenceTransformer model."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)


class CLIPTextEmbedder(Embedder):
    """Embed text using a CLIP model."""

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPTokenizerFast

        self.model = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)

    def embed(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model.get_text_features(**inputs)
        return outputs.detach().cpu().numpy()


class CLIPImageEmbedder(Embedder):
    """Embed images using a CLIP model."""

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        return outputs.detach().cpu().numpy()


class OpenAIEmbedder(Embedder):
    """Embed text using the OpenAI embeddings API."""

    def __init__(self, model_name="text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = model_name

    def embed(self, texts):
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return np.array([item.embedding for item in response.data])
