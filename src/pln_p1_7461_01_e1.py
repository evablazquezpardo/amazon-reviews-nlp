#!/usr/bin/env python3
"""pln_p1_7461_01_e1.py

Ejercicio 1: Descarga y preparación del corpus Home_and_Kitchen.
Lee el formato JSONL del dataset de Amazon Reviews '23 y genera documentos
JSON individuales en carpetas según polaridad (negative, neutral, positive).
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tempfile
from pathlib import Path
from typing import IO, Any, Iterator
from urllib.request import urlopen

# Categoría fijada para este ejercicio
CATEGORY = "Home_and_Kitchen"

DATA_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
REVIEW_URL = f"{DATA_URL}/review_categories/{CATEGORY}.jsonl.gz"
META_URL = f"{DATA_URL}/meta_categories/meta_{CATEGORY}.jsonl.gz"
POLARITIES = ("negative", "neutral", "positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga y procesa el corpus Amazon Reviews '23."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corpus_amazon"),
        help="Carpeta donde se guardará el corpus clasificado.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_raw"),
        help="Carpeta donde se encuentran (o descargarán) los archivos .jsonl/.jsonl.gz.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=5000,
        help="Número máximo de reseñas por polaridad (por defecto: 5000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribir el corpus generado si ya existe.",
    )
    return parser.parse_args()


def download_file(url: str, dest_path: Path) -> Path:
    """Descarga de forma segura un archivo temporalmente."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=dest_path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urlopen(url) as res:
                shutil.copyfileobj(res, tmp)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    try:
        tmp_path.replace(dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return dest_path


def get_local_or_download(data_dir: Path, rel_filename: str, url: str) -> Path:
    """Detecta si el archivo .jsonl o .jsonl.gz existe localmente"""
    plain = data_dir / rel_filename
    compressed = data_dir / f"{rel_filename}.gz"

    if plain.exists():
        return plain
    if compressed.exists():
        return compressed

    print(f"Descargando {url} ...")
    return download_file(url, compressed if url.endswith(".gz") else plain)


def open_jsonl(path: Path) -> IO[str]:
    """Abre el archivo tanto si está comprimido en gzip como en texto plano."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def read_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    """Lectura streaming línea a línea."""
    with open_jsonl(path) as fp:
        for line in fp:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                yield json.loads(line_str)
            except json.JSONDecodeError:
                continue


def assign_polarity(rating: Any) -> str:
    """Asigna la polaridad según la puntuación de la reseña."""
    score = float(rating)
    if score < 3.0:
        return "negative"
    if score == 3.0:
        return "neutral"
    return "positive"


def extract_needed_metadata(meta_path: Path, target_asins: set[str]) -> dict[str, dict[str, Any]]:
    """Carga solo los metadatos de los productos referenciados para proteger la RAM."""
    extracted = {}
    if not target_asins:
        return extracted

    print(f"Buscando metadatos para {len(target_asins)} parent_asin únicos...")
    for item in read_jsonl_records(meta_path):
        p_asin = str(item.get("parent_asin", ""))
        if p_asin in target_asins:
            item.pop("images", None)
            item.pop("videos", None)
            extracted[p_asin] = item
            if len(extracted) == len(target_asins):
                break
    return extracted


def main() -> None:
    args = parse_args()

    if args.max_per_class <= 0:
        raise ValueError("--max-per-class debe ser mayor que cero.")

    if args.overwrite and args.output.exists():
        shutil.rmtree(args.output)

    # 1. Crear directorios para CategorizedCorpusReader
    for pol in POLARITIES:
        (args.output / pol).mkdir(parents=True, exist_ok=True)

    # 2. Rutas y descarga de la categoría específica
    rev_rel = f"review_categories/{CATEGORY}.jsonl"
    meta_rel = f"meta_categories/meta_{CATEGORY}.jsonl"

    reviews_file = get_local_or_download(args.data_dir, rev_rel, REVIEW_URL)
    metadata_file = get_local_or_download(args.data_dir, meta_rel, META_URL)

    # 3. Leer como máximo el mismo número de reseñas por polaridad
    selected_reviews = {pol: [] for pol in POLARITIES}
    needed_asins = set()
    seen_user_product_texts: dict[tuple[str, str], set[str]] = {}

    total_lines_read = 0
    discarded_reviews = 0

    print(f"Leyendo reseñas desde {reviews_file.name}...")
    for raw_rev in read_jsonl_records(reviews_file):
        total_lines_read += 1

        title = str(raw_rev.get("title") or "").strip()
        body = str(raw_rev.get("text") or "").strip()

        if body and title:
            full_text = f"{title}. {body}"
        elif body:
            full_text = body
        elif title:
            full_text = title
        else:
            full_text = ""

        # Descarte si no contiene ningún texto
        if not full_text:
            discarded_reviews += 1
            continue

        u_id = str(raw_rev["user_id"]).strip()
        p_asin = str(raw_rev["parent_asin"]).strip()
        user_product_key = (u_id, p_asin)

        # Descarte si el mismo usuario ya escribió un texto idéntico para este producto
        if user_product_key in seen_user_product_texts:
            if full_text in seen_user_product_texts[user_product_key]:
                discarded_reviews += 1
                continue
            else:
                seen_user_product_texts[user_product_key].add(full_text)
        else:
            seen_user_product_texts[user_product_key] = {full_text}

        pol = assign_polarity(raw_rev["rating"])

        # Control de "cupo" por polaridad
        if len(selected_reviews[pol]) >= args.max_per_class:
            if all(
                len(selected_reviews[current_pol]) >= args.max_per_class
                for current_pol in POLARITIES
            ):
                break
            continue

        # ID único con timestamp para no pisar comentarios con textos distintos
        ts = str(raw_rev.get("timestamp") or "")
        unique_id = f"{u_id}_{p_asin}_{ts}" if ts else f"{u_id}_{p_asin}_{len(selected_reviews[pol])}"

        raw_rev["_full_text"] = full_text
        raw_rev["_unique_id"] = unique_id
        selected_reviews[pol].append(raw_rev)
        needed_asins.add(p_asin)

        if all(
            len(selected_reviews[current_pol]) >= args.max_per_class
            for current_pol in POLARITIES
        ):
            break

    # 4. Vincular metadatos de producto
    meta_dict = extract_needed_metadata(metadata_file, needed_asins)

    # 5. Generar ficheros JSON individuales por categoría/polaridad
    counts = {pol: 0 for pol in POLARITIES}
    print("Guardando archivos JSON en disco...")

    for pol in POLARITIES:
        for raw_rev in selected_reviews[pol]:
            counts[pol] += 1
            unique_id = raw_rev["_unique_id"]

            doc = {
                "id": unique_id,
                "category": CATEGORY,
                "polarity": pol,
                "rating": raw_rev.get("rating"),
                "title": raw_rev.get("title", ""),
                "text": raw_rev["_full_text"],
                "asin": raw_rev.get("asin"),
                "parent_asin": raw_rev.get("parent_asin"),
                "user_id": raw_rev.get("user_id"),
                "timestamp": raw_rev.get("timestamp"),
                "helpful_vote": raw_rev.get("helpful_vote", 0),
                "verified_purchase": raw_rev.get("verified_purchase", False),
                "product_metadata": meta_dict.get(str(raw_rev.get("parent_asin"))),
            }

            file_path = args.output / pol / f"{unique_id}.json"
            with open(file_path, "w", encoding="utf-8") as out_fp:
                json.dump(doc, out_fp, ensure_ascii=False, indent=2)

    # Resumen del corpus generado
    print(f"\nProceso finalizado:")
    print(f"- Líneas leídas: {total_lines_read}")
    print(f"- Eliminadas:   {discarded_reviews}")
    print(f"- Positivas:    {counts['positive']}")
    print(f"- Neutras:      {counts['neutral']}")
    print(f"- Negativas:    {counts['negative']}")
    print(f"- Directorio de salida: {args.output.resolve()}")


if __name__ == "__main__":
    main()