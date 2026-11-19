"""Exporta las sílabas de práctica de Tone Perfect al frontend.

Lee los XML de metadatos y guarda, por sílaba, los tonos en los que es una palabra real
con su pinyin y sus caracteres. Las combinaciones marcadas como hueco léxico (sílaba
grabada en un tono en el que no existe como palabra, p. ej. "bán") se omiten: practicar
sonidos que no son palabras aporta poco a un aprendiz.

El corpus escribe la ü como "v" (lv, nve); en el pinyin que se muestra se sustituye.

Uso: uv run python scripts/05_exportar_silabas_toneperfect.py
"""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--xml-dir", default="data/raw/toneperfect/tone_perfect_xml")
ap.add_argument("--out", default="frontend/src/data/syllables.json")
args = ap.parse_args()


def read_entry(path: Path) -> dict:
    root = ET.parse(path).getroot()
    return {
        "syllable": root.findtext("sound"),
        "tone": int(root.findtext("tone")),
        "pinyin": root.findtext("pinyin").replace("v", "ü"),
        "characters": [c.findtext("simplified") for c in root.findall("character")],
        "is_lexical_gap": root.findtext("lexical_gap") == "Yes",
    }


# Cada sílaba-tono tiene un XML por hablante con los mismos metadatos; se comprueba.
entries: dict[tuple[str, int], dict] = {}
for path in sorted(Path(args.xml_dir).glob("*_CUSTOM.xml")):
    entry = read_entry(path)
    key = (entry["syllable"], entry["tone"])
    if key in entries and entries[key] != entry:
        raise SystemExit(f"Metadatos distintos entre hablantes para {key}: {path.name}")
    entries[key] = entry

by_syllable: dict[str, dict[str, dict]] = defaultdict(dict)
for (syllable, tone), entry in entries.items():
    if not entry["is_lexical_gap"]:
        by_syllable[syllable][str(tone)] = {"pinyin": entry["pinyin"], "characters": entry["characters"]}

syllables = [
    {"syllable": syllable.replace("v", "ü"), "tones": dict(sorted(tones.items()))}
    for syllable, tones in sorted(by_syllable.items())
]
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(syllables, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
n_items = sum(len(s["tones"]) for s in syllables)
n_gaps = sum(e["is_lexical_gap"] for e in entries.values())
print(f"{len(syllables)} sílabas, {n_items} combinaciones sílaba-tono "
      f"({n_gaps} huecos léxicos omitidos) | guardado en {out}")
