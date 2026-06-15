# API reference

Auto-generated from the source docstrings. Everything below is also importable
directly from the top-level `splytters` namespace (e.g.
`from splytters import cluster_split`).

## Adversarial splitters

::: splytters.adversarial

## Overlap splitters

::: splytters.overlap

## Balanced splitters

::: splytters.balanced

## Curriculum splits

::: splytters.curriculum

## scikit-learn compatibility

::: splytters.sklearn_api

## Framework interop

::: splytters.interop

## Split-quality reporting

::: splytters.report

## Sorters

Ranking functions that order samples by interpretable difficulty/quality
metrics, grouped by modality. Pair a ranking with
[`sorted_stratified_split`](#curriculum-splits) for a curriculum split. Each
modality imports its heavy dependency lazily, so importing a sorter module needs
no optional extra.

### Embedding sorters

::: splytters.sorters.embedding_sorters

### Text sorters

::: splytters.sorters.text_sorters

### Image sorters

::: splytters.sorters.image_sorters

### Audio sorters

::: splytters.sorters.audio_sorters

### Tabular sorters

::: splytters.sorters.tabular_sorters

## Utilities

::: splytters.utils

## Embedders

::: splytters.embedders

## Introspection

::: splytters.list_splitters

::: splytters.sorters.list_sorters

::: splytters.embedders.list_embedders

## Types

::: splytters.Splitter
