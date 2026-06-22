"""Fast, dependency-free tests for the embedder wrappers.

These inject fake model libraries (sentence-transformers, transformers, openai)
via ``sys.modules`` so the thin wrapper logic — lazy import, construct, encode —
is exercised without downloading any model or importing torch. The real-model
counterparts live in ``test_embedders.py`` (marked slow).
"""

import sys
import types

import numpy as np

from splytters.embedders import (
    CLIPImageEmbedder,
    CLIPTextEmbedder,
    OpenAIEmbedder,
    TextEmbedder,
    _features_to_numpy,
    list_embedders,
)


def _fake_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class _FakeTensor:
    """Stands in for a torch tensor: supports .detach().cpu().numpy()."""

    def __init__(self, arr):
        self.arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


def test_list_embedders_includes_concrete_classes():
    names = list_embedders()
    assert {"TextEmbedder", "CLIPTextEmbedder", "CLIPImageEmbedder",
            "OpenAIEmbedder"} <= set(names)


class TestFeaturesToNumpy:
    """The three branches of the CLIP feature -> numpy converter."""

    def test_tensor_branch(self):
        out = _features_to_numpy(_FakeTensor(np.ones((2, 3))))
        assert out.shape == (2, 3)

    def test_pooler_output_branch(self):
        outputs = types.SimpleNamespace(pooler_output=_FakeTensor(np.ones((2, 4))))
        assert _features_to_numpy(outputs).shape == (2, 4)

    def test_last_hidden_state_branch(self):
        lhs = types.SimpleNamespace(mean=lambda dim: _FakeTensor(np.ones((2, 5))))
        outputs = types.SimpleNamespace(last_hidden_state=lhs)
        assert _features_to_numpy(outputs).shape == (2, 5)


def test_text_embedder(monkeypatch):
    class FakeST:
        def __init__(self, name):
            self.name = name

        def encode(self, texts, **kw):
            return np.random.RandomState(0).rand(len(texts), 4)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        _fake_module("sentence_transformers", SentenceTransformer=FakeST),
    )
    out = TextEmbedder().embed(["a", "b", "c"])
    assert out.shape == (3, 4)


def test_clip_text_embedder(monkeypatch):
    arr = np.ones((2, 5))

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def get_text_features(self, **inputs):
            return _FakeTensor(arr)

    class FakeTok:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def __call__(self, texts, **kw):
            return {"input_ids": [[0]]}

    monkeypatch.setitem(
        sys.modules, "transformers",
        _fake_module("transformers", CLIPModel=FakeModel, CLIPTokenizerFast=FakeTok),
    )
    out = CLIPTextEmbedder().embed(["a", "b"])
    assert out.shape == (2, 5)


def test_clip_image_embedder(monkeypatch):
    arr = np.ones((2, 6))

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def get_image_features(self, **inputs):
            return _FakeTensor(arr)

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def __call__(self, images, **kw):
            return {"pixel_values": [[0]]}

    monkeypatch.setitem(
        sys.modules, "transformers",
        _fake_module("transformers", CLIPModel=FakeModel, CLIPProcessor=FakeProcessor),
    )
    out = CLIPImageEmbedder().embed(["img1", "img2"])
    assert out.shape == (2, 6)


def test_openai_embedder(monkeypatch):
    class FakeEmbeddings:
        def create(self, input, model):
            data = [types.SimpleNamespace(embedding=[0.1, 0.2, 0.3, 0.4])
                    for _ in input]
            return types.SimpleNamespace(data=data)

    class FakeOpenAI:
        def __init__(self):
            self.embeddings = FakeEmbeddings()

    monkeypatch.setitem(sys.modules, "openai",
                        _fake_module("openai", OpenAI=FakeOpenAI))
    out = OpenAIEmbedder().embed(["a", "b"])
    assert out.shape == (2, 4)
