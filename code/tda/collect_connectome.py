"""
Descarga el conectoma de C. elegans desde WormAtlas.

Convierte el archivo NeuronConnect.xls en un CSV limpio con
las columnas: neuron1, neuron2, type, weight.

Requisitos: pip install openpyxl requests
Salida: data/elegans_connectome.csv
"""

import csv
import os
import ssl
import sys

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = "https://www.wormatlas.org/images/NeuronConnect.xls"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "elegans_connectome.csv")


def download_xls(url=URL):
    """Descarga el archivo XLS de WormAtlas."""
    print(f"Descargando {url} ...")
    resp = requests.get(url, timeout=30, verify=False)
    resp.raise_for_status()
    return resp.content


def xls_to_rows(xls_bytes):
    """Lee el XLS y extrae filas (neuron1, neuron2, type, weight)."""
    import xlrd
    import tempfile

    # xlrd needs a file path, not bytes
    with tempfile.NamedTemporaryFile(suffix=".xls", delete=False) as tmp:
        tmp.write(xls_bytes)
        tmp_path = tmp.name

    try:
        wb = xlrd.open_workbook(tmp_path)
        ws = wb.sheet_by_index(0)

        rows = []
        for i in range(1, ws.nrows):
            n1 = str(ws.cell_value(i, 0)).strip()
            n2 = str(ws.cell_value(i, 1)).strip()
            syn_type = str(ws.cell_value(i, 2)).strip()
            w = ws.cell_value(i, 3)
            weight = int(w) if w else 1
            if n1:
                rows.append((n1, n2, syn_type, weight))
        return rows
    finally:
        os.remove(tmp_path)


def save_csv(rows, path=OUTPUT_CSV):
    """Guarda las filas como CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["neuron1", "neuron2", "type", "weight"])
        writer.writerows(rows)
    print(f"Guardado: {path} ({len(rows)} conexiones)")


def main():
    xls_bytes = download_xls()
    rows = xls_to_rows(xls_bytes)
    save_csv(rows)
    print(f"Tipos de sinapsis: {sorted(set(r[2] for r in rows))}")
    print(f"Neuronas únicas: {len(set(r[0] for r in rows) | set(r[1] for r in rows))}")


if __name__ == "__main__":
    main()
