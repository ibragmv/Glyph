from __future__ import annotations

from pathlib import Path


IMPERIAL_ARAMAIC_SYMBOLS = [
    {"index": 0, "name": "aleph", "codepoint": 0x10840, "char": "\U00010840"},
    {"index": 1, "name": "beth", "codepoint": 0x10841, "char": "\U00010841"},
    {"index": 2, "name": "gimel", "codepoint": 0x10842, "char": "\U00010842"},
    {"index": 3, "name": "daleth", "codepoint": 0x10843, "char": "\U00010843"},
    {"index": 4, "name": "he", "codepoint": 0x10844, "char": "\U00010844"},
    {"index": 5, "name": "waw", "codepoint": 0x10845, "char": "\U00010845"},
    {"index": 6, "name": "zayin", "codepoint": 0x10846, "char": "\U00010846"},
    {"index": 7, "name": "heth", "codepoint": 0x10847, "char": "\U00010847"},
    {"index": 8, "name": "teth", "codepoint": 0x10848, "char": "\U00010848"},
    {"index": 9, "name": "yodh", "codepoint": 0x10849, "char": "\U00010849"},
    {"index": 10, "name": "kaph", "codepoint": 0x1084A, "char": "\U0001084A"},
    {"index": 11, "name": "lamedh", "codepoint": 0x1084B, "char": "\U0001084B"},
    {"index": 12, "name": "mem", "codepoint": 0x1084C, "char": "\U0001084C"},
    {"index": 13, "name": "nun", "codepoint": 0x1084D, "char": "\U0001084D"},
    {"index": 14, "name": "samekh", "codepoint": 0x1084E, "char": "\U0001084E"},
    {"index": 15, "name": "ayin", "codepoint": 0x1084F, "char": "\U0001084F"},
    {"index": 16, "name": "pe", "codepoint": 0x10850, "char": "\U00010850"},
    {"index": 17, "name": "sadhe", "codepoint": 0x10851, "char": "\U00010851"},
    {"index": 18, "name": "qoph", "codepoint": 0x10852, "char": "\U00010852"},
    {"index": 19, "name": "resh", "codepoint": 0x10853, "char": "\U00010853"},
    {"index": 20, "name": "shin", "codepoint": 0x10854, "char": "\U00010854"},
    {"index": 21, "name": "taw", "codepoint": 0x10855, "char": "\U00010855"},
]

LABEL_DIRS = {
    symbol["index"]: f"{symbol['index']:02d}_{symbol['name']}"
    for symbol in IMPERIAL_ARAMAIC_SYMBOLS
}
CLASS_NAMES = [symbol["name"] for symbol in IMPERIAL_ARAMAIC_SYMBOLS]
CODEPOINTS = [symbol["codepoint"] for symbol in IMPERIAL_ARAMAIC_SYMBOLS]

DEFAULT_FONT_SEARCH_PATHS = [
    Path("fonts"),
    Path("/System/Library/Fonts/Supplemental"),
    Path("/Library/Fonts"),
    Path.home() / "Library" / "Fonts",
]

DEFAULT_FONT_CANDIDATES = [
    "NotoSansImperialAramaic-Regular.ttf",
    "NotoSansImperialAramaic.ttf",
]
