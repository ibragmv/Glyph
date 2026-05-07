from __future__ import annotations

from pathlib import Path


CLASS_TITLES = [
    "Alef",
    "Bet",
    "Gimel",
    "Dalet",
    "He",
    "Waw",
    "Zayin",
    "Chet",
    "Tet",
    "Yod",
    "Kaf",
    "Lamed",
    "Mem",
    "Nun",
    "Samech",
    "Ayin",
    "Pe",
    "Tsadik",
    "Qof",
    "Resh",
    "Shin",
    "Tav",
]

CLASS_NAMES = [title.lower() for title in CLASS_TITLES]

ALPHABET = [
    {
        "index": index,
        "name": name,
        "title": title,
        "label_dir": f"{index:02d}_{name}",
    }
    for index, (name, title) in enumerate(zip(CLASS_NAMES, CLASS_TITLES))
]

LABEL_DIRS = {item["index"]: item["label_dir"] for item in ALPHABET}
TITLE_BY_NAME = {item["name"]: item["title"] for item in ALPHABET}
NAME_BY_TITLE = {item["title"]: item["name"] for item in ALPHABET}

SOURCE_ROOT = Path("source")
ALPHABET_ROOT = SOURCE_ROOT / "alphabet"
EXEMPLAR_ROOT = SOURCE_ROOT / "exemplars"
REAL_ROOT = SOURCE_ROOT / "real"
TEXTURE_ROOT = SOURCE_ROOT / "textures"

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
R_TRAIN_SPLIT = "r_train"
R_VAL_SPLIT = "r_val"
