"""Liste figée de photos NASA (domaine public, Wikimedia Commons) pour le banc d'essai des détecteurs.

Depuis la racine du dépôt :
    python scripts/nasa_photos.py --build      # interroge Commons et écrit docs/nasa_photos.csv
    python scripts/nasa_photos.py --download   # télécharge les vignettes absentes dans data/demo/nasa/
"""

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "FER2013-educational-demo/1.0 (projet étudiant ING3, usage pédagogique)"
MANIFEST = Path("docs/nasa_photos.csv")
PHOTO_DIR = Path("data/demo/nasa")
THUMB_WIDTH = 1600
FIELDS = ("fichier", "source", "titre_commons", "page_commons", "url_vignette", "licence",
          "conditions", "credit", "auteur", "date", "description")
# Source -> (recherches Commons, nombre maximal de photos retenues).
SOURCES = {
    "portrait_officiel": (['intitle:"crew portrait" intitle:Expedition'], 50),
    "en_orbite": ([
        'intitle:"pose for a portrait" "International Space Station"',
        'intitle:"gather for a portrait" "International Space Station"',
        'intitle:"gathers for a portrait" "International Space Station"',
        'intitle:"crew members" intitle:portrait intitle:"International Space Station"',
    ], 20),
    "artemis": ([
        'intitle:"Artemis 2 Crew Portrait"',
        'intitle:"Artemis II Crew" intitle:"Rise"',
        'intitle:"Artemis II Launch Crew Walkout"',
        'intitle:"Artemis II Crew Arrives at KSC"',
        'intitle:"Artemis II Crew Q and A"',
    ], 10),
}
EXCLUDED = re.compile(r"cropped|visuali[sz]ation|screenshot|patch|logo|insignia", re.IGNORECASE)
NASA_PATTERN = re.compile(r"nasa|national aeronautics|space center", re.IGNORECASE)


def _get(params: dict) -> dict:
    """Requête API temporisée, avec nouvelle tentative après un refus pour débit trop élevé."""
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(5):
        time.sleep(1.0)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 4:
                raise
            time.sleep(int(error.headers.get("Retry-After", 10)))
    raise RuntimeError("Commons indisponible.")


def _text(meta: dict, key: str) -> str:
    """Valeur extmetadata sans balises HTML ni espaces superflus."""
    value = re.sub(r"<[^>]+>", " ", str(meta.get(key, {}).get("value", "")))
    return re.sub(r"\s+", " ", value).strip()


def _search(query: str) -> list[str]:
    """Titres de fichiers Commons correspondant à une recherche."""
    titles, offset = [], 0
    while True:
        data = _get({"action": "query", "list": "search", "srnamespace": 6, "srlimit": 50,
                     "sroffset": offset, "srsearch": query})
        titles += [result["title"] for result in data["query"]["search"]]
        if "continue" not in data or len(titles) >= 200:
            return titles
        offset = data["continue"]["sroffset"]


def _rows(source: str, titles: list[str]) -> list[dict]:
    """Métadonnées des fichiers retenus : JPEG, domaine public, crédit NASA, largeur suffisante."""
    rows = []
    for start in range(0, len(titles), 50):
        data = _get({"action": "query", "prop": "imageinfo", "titles": "|".join(titles[start:start + 50]),
                     "iiprop": "url|size|mime|extmetadata", "iiurlwidth": THUMB_WIDTH})
        for page in data["query"]["pages"].values():
            info = page.get("imageinfo", [{}])[0]
            meta = info.get("extmetadata", {})
            credit, author = _text(meta, "Credit"), _text(meta, "Artist")
            if (info.get("mime") != "image/jpeg" or _text(meta, "LicenseShortName") != "Public domain"
                    or not NASA_PATTERN.search(credit + " " + author) or info.get("width", 0) < 1000
                    or EXCLUDED.search(page["title"])):
                continue
            rows.append({
                "source": source, "titre_commons": page["title"],
                "page_commons": info["descriptionurl"], "url_vignette": info["thumburl"],
                "licence": _text(meta, "LicenseShortName"), "conditions": _text(meta, "UsageTerms"),
                "credit": credit[:300], "auteur": author[:200], "date": _text(meta, "DateTimeOriginal")[:40],
                "description": _text(meta, "ImageDescription")[:300],
            })
    return sorted(rows, key=lambda row: row["titre_commons"])


def build(manifest: Path = MANIFEST) -> list[dict]:
    """Interroge Commons et écrit la liste figée des photos, avec licence et crédit."""
    selected, seen = [], set()
    for source, (queries, limit) in SOURCES.items():
        titles = list(dict.fromkeys(title for query in queries for title in _search(query)))
        rows = [row for row in _rows(source, titles) if row["titre_commons"] not in seen]
        rows = rows[::max(1, len(rows) // limit)][:limit]  # Échantillon régulier dans l'ordre alphabétique.
        seen.update(row["titre_commons"] for row in rows)
        selected += rows
        print(f"{source} : {len(rows)} photos retenues sur {len(titles)} trouvées.")
    for index, row in enumerate(selected, start=1):  # Numéro en tête : noms uniques et ordre stable.
        stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(row["titre_commons"].removeprefix("File:")).stem)
        row["fichier"] = f"{index:03d}_{stem.strip('_')[:60]}.jpg"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    print(f"{len(selected)} photos dans {manifest}.")
    return selected


def download(manifest: Path = MANIFEST, photo_dir: Path = PHOTO_DIR) -> None:
    """Télécharge les vignettes absentes, sans écraser les fichiers présents."""
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    photo_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = photo_dir / row["fichier"]
        if path.exists():
            continue
        time.sleep(0.5)
        request = urllib.request.Request(row["url_vignette"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response, path.open("xb") as output:
            output.write(response.read())
    print(f"{sum((photo_dir / row['fichier']).exists() for row in rows)}/{len(rows)} photos dans {photo_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true", help="Reconstruire docs/nasa_photos.csv depuis Commons.")
    parser.add_argument("--download", action="store_true", help="Télécharger les vignettes absentes.")
    args = parser.parse_args()
    if args.build:
        build()
    if args.download:
        download()
