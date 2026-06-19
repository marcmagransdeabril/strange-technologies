"""
Descarga libros clásicos de Project Gutenberg, los segmenta
en ventanas de texto, y calcula embeddings con sentence-transformers.

Genera un archivo NPZ con:
  - embeddings:    (N, 384) float32
  - positions:     (N,) float32  — posición narrativa normalizada [0, 1]
  - book_ids:      (N,) int32    — índice del libro
  - book_names:    lista de nombres
  - has_location:  (N,) bool     — True si el chunk menciona un lugar (NER)
  - locations:     (N,) object   — lista de nombres de lugar por chunk (NER)
  - location_count: (N,) int32   — número de entidades de lugar por chunk

Requisitos: pip install requests sentence-transformers spacy
            python -m spacy download en_core_web_sm
Salida: data/hero_books.npz
"""

import os
import re
import ssl
import sys

import numpy as np
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_NPZ = os.path.join(OUTPUT_DIR, "hero_books.npz")

# Gutenberg plain-text mirrors (UTF-8)
BOOKS = [
    {"name": "The Odyssey", "id": 1727},
    {"name": "Beowulf", "id": 981},
    {"name": "Divine Comedy", "id": 8800},
    {"name": "Don Quixote", "id": 996},
    {"name": "Moby Dick", "id": 2701},
    {"name": "Monte Cristo", "id": 1184},
    {"name": "Wizard of Oz", "id": 55},
    {"name": "Around the World 80 Days", "id": 103},
    {"name": "Princess of Mars", "id": 62},
]

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
WINDOW_WORDS = 500
STRIDE_WORDS = 250  # 50% overlap


def download_text(ebook_id):
    """Descarga el texto plano de Gutenberg."""
    url = GUTENBERG_URL.format(ebook_id=ebook_id)
    print(f"  Descargando {url} ...")
    resp = requests.get(url, timeout=60, verify=False)
    resp.raise_for_status()
    return resp.text


def strip_gutenberg_header_footer(text):
    """Elimina cabecera y pie de Project Gutenberg."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Move past the marker line
            start_idx = text.find("\n", idx) + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def chunk_text(text, window_words=WINDOW_WORDS, stride_words=STRIDE_WORDS):
    """Divide el texto en ventanas solapadas de N palabras."""
    words = text.split()
    chunks = []
    positions = []
    total = len(words)
    i = 0
    while i < total:
        end = min(i + window_words, total)
        chunk = " ".join(words[i:end])
        if len(chunk.strip()) > 50:  # skip tiny final chunks
            chunks.append(chunk)
            mid = (i + end) / 2.0
            positions.append(mid / total)  # normalized [0, 1]
        i += stride_words
    return chunks, positions


def compute_embeddings(chunks, model_name="all-MiniLM-L6-v2", batch_size=64):
    """Calcula embeddings con sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print(f"  Calculando embeddings ({len(chunks)} ventanas) ...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        chunks, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def detect_locations(chunks, nlp):
    """Detecta entidades de lugar en cada chunk (NER).

    Returns:
        flags: (N,) bool — True si el chunk menciona al menos un lugar
        loc_names: list of lists — nombres de lugar únicos por chunk
        loc_counts: (N,) int — número de entidades de lugar por chunk
    """
    flags = []
    loc_names = []
    loc_counts = []
    for doc in nlp.pipe(chunks, batch_size=64):
        locs = list({
            ent.text for ent in doc.ents
            if ent.label_ in ("GPE", "LOC", "FAC")
        })
        flags.append(len(locs) > 0)
        loc_names.append(locs)
        loc_counts.append(len(locs))
    return (
        np.array(flags, dtype=bool),
        loc_names,
        np.array(loc_counts, dtype=np.int32),
    )


def main():
    all_embeddings = []
    all_positions = []
    all_book_ids = []
    all_has_location = []
    all_locations = []
    all_location_counts = []
    book_names = []

    model = None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        print("ERROR: sentence-transformers no instalado")
        print("  pip install sentence-transformers")
        sys.exit(1)

    nlp = None
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
    except (ImportError, OSError) as e:
        print(f"WARN: spaCy no disponible ({e}), has_location será False")
        print("  pip install spacy && python -m spacy download en_core_web_sm")

    for i, book in enumerate(BOOKS):
        print(f"\n[{i+1}/{len(BOOKS)}] {book['name']} (PG #{book['id']})")
        text = download_text(book["id"])
        text = strip_gutenberg_header_footer(text)
        chunks, positions = chunk_text(text)
        print(f"  {len(chunks)} ventanas de texto")

        embeddings = model.encode(
            chunks, batch_size=64, show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Detect location entities in each chunk
        if nlp is not None:
            print(f"  Detectando localizaciones (NER) ...")
            loc_flags, loc_names, loc_counts = detect_locations(chunks, nlp)
        else:
            loc_flags = np.zeros(len(chunks), dtype=bool)
            loc_names = [[] for _ in chunks]
            loc_counts = np.zeros(len(chunks), dtype=np.int32)

        all_embeddings.append(embeddings)
        all_positions.extend(positions)
        all_book_ids.extend([i] * len(chunks))
        all_has_location.extend(loc_flags)
        all_locations.extend(loc_names)
        all_location_counts.extend(loc_counts)
        book_names.append(book["name"])

    embeddings = np.vstack(all_embeddings)
    positions = np.array(all_positions, dtype=np.float32)
    book_ids = np.array(all_book_ids, dtype=np.int32)
    has_location = np.array(all_has_location, dtype=bool)
    locations = np.array(all_locations, dtype=object)
    location_count = np.array(all_location_counts, dtype=np.int32)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        embeddings=embeddings,
        positions=positions,
        book_ids=book_ids,
        book_names=book_names,
        has_location=has_location,
        locations=locations,
        location_count=location_count,
    )
    size_mb = os.path.getsize(OUTPUT_NPZ) / 1024 / 1024
    print(f"\nGuardado: {OUTPUT_NPZ}")
    print(f"  Forma embeddings: {embeddings.shape}")
    print(f"  Chunks con localizaciones: {has_location.sum()}/{len(has_location)}")
    print(f"  Libros: {book_names}")
    print(f"  Tamaño: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
