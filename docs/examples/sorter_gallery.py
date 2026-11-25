"""Generate docs/sorter_gallery.md: every sorter applied to a small 10-item example.

Run from the repository root:

    python docs/examples/sorter_gallery.py

Needs the text, image, audio, tabular and embedders extras
(``pip install -e ".[all]"``), NLTK's ``punkt`` data for ``readability_score``,
and network access for the first download of all-MiniLM-L6-v2 and GPT-2.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from splytters import sorters as S
from splytters.embedders import TextEmbedder

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "test_data"
OUT = ROOT / "docs" / "sorter_gallery.md"
ASSETS = ROOT / "docs" / "assets" / "gallery"

QUERIES = [
    "what is my balance",
    "my bank balance is what",
    "balance",
    "what's my balance",
    "tell me my balance please",
    "can you check how much money is in my checking account right now",
    "what is the current balance of my savings account at the bank",
    "show me my balance and then show me my balance history",
    "how much cash do i have",
    "how full are my coffers today",
]
IMAGES = ["brightness_dark", "brightness_bright", "color_grayscale", "color_vibrant",
          "compress_simple", "compress_complex", "contrast_low", "contrast_high",
          "freq_smooth", "freq_detailed"]
AUDIO = ["loudness_quiet", "loudness_loud", "pitch_low", "pitch_high", "tempo_slow",
         "tempo_fast", "spectral_dark", "spectral_bright", "flatness_tonal", "flatness_noisy"]


def fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    if np.isinf(v):
        return "inf"
    if np.isnan(v):
        return "nan"
    return f"{v:.3g}"


def df_markdown(df: pd.DataFrame) -> str:
    rows = ["| | " + " | ".join(df.columns) + " |", "|---|" + "---|" * len(df.columns)]
    for idx, r in df.iterrows():
        rows.append(f"| {idx} | " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(rows)


def order_row(name: str, ranked, label) -> str:
    cells = " → ".join(f"{label(i)} ({fmt(s)})" for i, s in ranked)
    return f"| `{name}` | {cells} |"


def text_section(lines: list[str]) -> None:
    lines += ["## Text sorters", "",
              "The 10 example queries all ask for an account balance:", "",
              "| ID | Query |", "|---|---|"]
    lines += [f"| Q{i + 1} | {q} |" for i, q in enumerate(QUERIES)]
    lines += ["",
              "Each row lists the queries from the lowest to the highest score (in parentheses).",
              "", "| Sorter | Order, lowest score first |", "|---|---|"]
    for name in S.list_sorters(by_modality=True)["text"]:
        ranked = getattr(S, name)(list(QUERIES))
        lines.append(order_row(name, ranked, lambda i: f"Q{i + 1}"))
    lines += ["", "`readability_score` requires at least 100 words per text, so every query "
              "scores `inf` and keeps its input order. Use it on paragraphs or documents.", ""]


def embedding_section(lines: list[str]) -> None:
    X = TextEmbedder("all-MiniLM-L6-v2").embed(QUERIES)
    lines += ["## Embedding sorters", "",
              "The same 10 queries, embedded with `TextEmbedder(\"all-MiniLM-L6-v2\")`.", "",
              "| Sorter | Order, lowest score first |", "|---|---|"]
    for name in S.list_sorters(by_modality=True)["embedding"]:
        fn = getattr(S, name)
        if name == "knn_label_disagreement":
            continue
        lines.append(order_row(name, fn(X), lambda i: f"Q{i + 1}"))
    lines += ["", "`knn_label_disagreement` is not shown here because it also requires class "
              "labels. It ranks each sample by the fraction of its nearest neighbors that have a "
              "different label. With 10 samples in 384 dimensions, the sample covariance is "
              "singular, so `mahalanobis_distance_to_mean` uses the Ledoit-Wolf shrinkage "
              "estimate of the covariance instead.", ""]


def image_section(lines: list[str]) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in IMAGES:
        src = DATA / "images" / f"{name}.png"
        shutil.copyfile(src, ASSETS / src.name)
        paths.append(str(src))

    def thumb(i: int) -> str:
        # Markdown image syntax (not raw <img>), so that MkDocs rewrites the relative
        # path for the page's URL. The width needs the attr_list extension.
        return f'![{IMAGES[i]}](assets/gallery/{IMAGES[i]}.png){{ width="40" title="{IMAGES[i]}" }}'

    lines += ["## Image sorters", "",
              "Ten synthetic test images from `test_data/images/`, named for the property "
              "that each one illustrates:", "",
              "| " + " | ".join(IMAGES) + " |", "|" + "---|" * len(IMAGES),
              "| " + " | ".join(thumb(i) for i in range(len(IMAGES))) + " |", "",
              "Each row shows the images from the lowest to the highest score "
              "(hover for the file name).", "",
              "| Sorter | Order, lowest score first |", "|---|---|"]
    for name in S.list_sorters(by_modality=True)["image"]:
        ranked = getattr(S, name)(list(paths))
        lines.append(f"| `{name}` | " + " ".join(thumb(i) for i, _ in ranked) + " |")
    lines.append("")


def audio_section(lines: list[str]) -> None:
    paths = [str(DATA / "audio" / f"{n}.wav") for n in AUDIO]
    lines += ["## Audio sorters", "",
              "Ten synthetic clips from `test_data/audio/`, named for the property that each "
              "one illustrates.", "",
              "| Sorter | Order, lowest score first |", "|---|---|"]
    for name in S.list_sorters(by_modality=True)["audio"]:
        ranked = getattr(S, name)(list(paths))
        lines.append(order_row(name, ranked, lambda i: f"`{AUDIO[i]}`"))
    lines.append("")


def tabular_section(lines: list[str]) -> None:
    df = pd.DataFrame({
        "age": [34, 29, 41, 38, 25, 52, 31, 47, 36, 90],
        "income": [52000, 48000, 61000, 0, 45000, 75000, np.nan, 66000, 58000, 950000],
        "city": ["Boston", "Boston", "Denver", "Boston", "Denver",
                 "Boston", "Denver", "Boston", "Boise", "Boston"],
        "plan": ["basic", "basic", "pro", "basic", "basic", "pro", np.nan, "pro", "basic", "pro"],
        "visits": [3, 5, 0, 4, 6, 2, 0, 3, 4, 1],
    }, index=[f"R{i + 1}" for i in range(10)])
    lines += ["## Tabular sorters", "",
              "A 10-row table with an outlier (R10), two missing values (R7) and a rare "
              "city (R9):", "",
              df_markdown(df), "",
              "Sorters that rank by one column use `income` (numeric) or `city` "
              "(categorical).", "",
              "| Sorter | Order, lowest score first |", "|---|---|"]
    column_args = {"categorical_rarity": {"column": "city"},
                   "column_absolute_zscore": {"column": "income"},
                   "column_rank": {"column": "income"}, "column_value": {"column": "income"},
                   "column_zscore": {"column": "income"},
                   "multi_column_sort": {"columns": ["income", "visits"], "weights": [1.0, 1.0]}}
    for name in S.list_sorters(by_modality=True)["tabular"]:
        ranked = getattr(S, name)(df, **column_args.get(name, {}))
        lines.append(order_row(name, ranked, lambda i: str(i)))
    lines.append("")


def main() -> None:
    lines = ["# Sorter gallery", "",
             "This page applies every sorter in `splytters.sorters` to a small example, so that "
             "you can see what each one ranks first and last. All sorters rank from the lowest "
             "to the highest score by default (`low_first=True`), and "
             "`sorted_stratified_split` assigns the first `train_size` fraction of each class "
             "to train, so by default the highest-scoring samples go to test.", "",
             "This page is generated by `docs/examples/sorter_gallery.py`.", ""]
    sections = (text_section, embedding_section, image_section, audio_section, tabular_section)
    for section in sections:
        section(lines)
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
