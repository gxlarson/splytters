"""Tests for the splitter/sorter introspection helpers."""

import splytters
import splytters.sorters as sorters

SPLIT_FAMILIES = {"adversarial", "overlap", "balanced", "baseline", "grouped"}
SORTER_MODALITIES = {"embedding", "text", "image", "audio", "tabular"}


# --- list_splitters ---------------------------------------------------------


def test_list_splitters_flat_is_callable_and_exported():
    names = splytters.list_splitters()
    assert isinstance(names, list)
    assert names, "expected at least one splitter"
    for name in names:
        assert name in splytters.__all__
        assert callable(getattr(splytters, name))


def test_list_splitters_by_family_partitions_the_flat_list():
    grouped = splytters.list_splitters(by_family=True)
    assert set(grouped) == SPLIT_FAMILIES

    # Families are disjoint and their union is exactly the flat list.
    flat = splytters.list_splitters()
    union = [name for names in grouped.values() for name in names]
    assert union == flat
    assert len(set(union)) == len(union), "a splitter appears in two families"


def test_list_splitters_covers_every_registered_split():
    """Drift guard: every public ``*_split`` defined in the adversarial/
    overlap/balanced modules (plus the baseline ``random_split``) must be
    registered in a family. Catches a new splitter added but not listed.

    Excludes the ``get_cluster_info`` helper (no ``_split`` suffix), the
    ``optimized_split`` utility and ``*_train_test_split`` sklearn wrappers
    (defined in other modules)."""
    family_modules = {
        "splytters.adversarial",
        "splytters.overlap",
        "splytters.balanced",
        "splytters.grouped",
    }
    actual = {
        n
        for n in splytters.__all__
        if n.endswith("_split")
        and getattr(getattr(splytters, n), "__module__", "") in family_modules
    }
    actual.add("random_split")  # baseline; lives in splytters.utils
    assert set(splytters.list_splitters()) == actual


def test_list_splitters_returns_copies():
    """Mutating the result must not corrupt the internal registry."""
    grouped = splytters.list_splitters(by_family=True)
    grouped["adversarial"].append("bogus")
    assert "bogus" not in splytters.list_splitters()


# --- list_sorters -----------------------------------------------------------


def test_list_sorters_flat_is_sorted_and_unique():
    names = sorters.list_sorters()
    assert isinstance(names, list)
    assert names == sorted(names)
    assert len(set(names)) == len(names)


def test_list_sorters_reexported_at_top_level():
    assert splytters.list_sorters() == sorters.list_sorters()


def test_list_sorters_by_modality_partitions_the_flat_list():
    grouped = sorters.list_sorters(by_modality=True)
    assert set(grouped) == SORTER_MODALITIES

    flat = set(sorters.list_sorters())
    union = [name for names in grouped.values() for name in names]
    assert set(union) == flat
    assert len(set(union)) == len(union), "a sorter appears in two modalities"
    for names in grouped.values():
        assert names == sorted(names)


def test_list_sorters_does_not_import_optional_deps():
    """Listing must not trigger the lazy modality imports."""
    import sys

    for heavy in ("librosa", "transformers", "PIL", "pandas"):
        sys.modules.pop(heavy, None)
    sorters.list_sorters(by_modality=True)
    for heavy in ("librosa", "transformers", "PIL", "pandas"):
        assert heavy not in sys.modules, f"listing should not import {heavy}"


# --- list_embedders ---------------------------------------------------------


def test_list_embedders_lists_concrete_classes():
    import splytters.embedders as embedders

    names = splytters.list_embedders()
    assert names == embedders.list_embedders()
    # The abstract base must not appear; the concrete embedders must.
    assert "Embedder" not in names
    assert {"TextEmbedder", "CLIPTextEmbedder", "CLIPImageEmbedder", "OpenAIEmbedder"} <= set(
        names
    )
    for name in names:
        assert issubclass(getattr(embedders, name), embedders.Embedder)


def test_list_embedders_does_not_import_optional_deps():
    import sys

    for heavy in ("sentence_transformers", "transformers", "openai"):
        sys.modules.pop(heavy, None)
    splytters.list_embedders()
    for heavy in ("sentence_transformers", "transformers", "openai"):
        assert heavy not in sys.modules, f"listing should not import {heavy}"


# --- Splitter type ----------------------------------------------------------


def test_splitter_type_is_exported_and_shared():
    """The Splitter alias is a single object re-used everywhere."""
    from splytters import Splitter, interop, report, sklearn_api

    assert Splitter is interop.Splitter
    assert Splitter is sklearn_api.Splitter
    assert Splitter is report.Splitter
