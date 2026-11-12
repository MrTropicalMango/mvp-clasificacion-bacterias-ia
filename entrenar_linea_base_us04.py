# UNIR
# Seminario de Innovación en Inteligencia Artificial
# Equipo 3013B
# Integrantes: Durán García Sergio Israel; Mallorquín Cabral Christian Javier; Mejía Sandoval Octavio Francisco; Velásquez Irías Fausto Rafael

"""US-04: línea base reproducible para el piloto de cinco especies.

Modelo: clasificador de centroides por clase. Cada microfotografía se reduce a
64x64 RGB y se compara con el promedio de las imágenes de entrenamiento de
cada especie. No pretende sustituir una CNN clínica; establece una referencia
simple, trazable y reproducible contra la cual comparar modelos posteriores.

Ejecutar (Google Colab o entorno con numpy y Pillow):
    python entrenar_linea_base_us04.py
"""

from __future__ import annotations

import csv
import os
import json
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "inventario_dataset_piloto" / "subconjunto_piloto.csv"
ZIP_DIR = Path(os.environ.get("MVP_DATA_DIR", ROOT / "dataset_piloto_bruto"))
OUT_DIR = Path(os.environ.get("MVP_BASELINE_OUTPUT", ROOT / "resultados_us04_linea_base"))
IMAGE_SIZE = (64, 64)
FOLDS = (1, 2, 3, 4, 5)


def load_records():
    with MANIFEST.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def vector_from_zip(record, archives):
    archive_name = record["archivo_zip"]
    if archive_name not in archives:
        archives[archive_name] = zipfile.ZipFile(ZIP_DIR / archive_name)
    data = archives[archive_name].read(record["ruta_en_zip"])
    with Image.open(BytesIO(data)) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0


def class_metrics(labels, truth, prediction):
    result = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, prediction))
        fp = sum(t != label and p == label for t, p in zip(truth, prediction))
        fn = sum(t == label and p != label for t, p in zip(truth, prediction))
        support = sum(t == label for t in truth)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "support": support,
        }
    return result


def write_csv(path, rows, headers):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_svg(path, labels, matrix):
    cell, left, top = 150, 260, 100
    width, height = left + cell * len(labels) + 30, top + cell * len(labels) + 90
    max_value = max(max(row) for row in matrix) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<style>text{font-family:Arial,sans-serif;font-size:13px}.title{font-size:18px;font-weight:bold}</style>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="28" class="title">US-04 — Matriz de confusión acumulada (CV-5)</text>',
             '<text x="20" y="53">Filas: clase real. Columnas: clase predicha. Línea base: centroide RGB 64×64.</text>']
    for j, label in enumerate(labels):
        parts.append(f'<text x="{left + j*cell + cell/2}" y="88" text-anchor="middle">{label}</text>')
    for i, label in enumerate(labels):
        y = top + i * cell
        parts.append(f'<text x="{left-8}" y="{y+cell/2}" text-anchor="end">{label}</text>')
        for j, value in enumerate(matrix[i]):
            blue = int(245 - 130 * value / max_value)
            parts.append(f'<rect x="{left+j*cell}" y="{y}" width="{cell}" height="{cell}" fill="rgb({blue},{blue+5},255)" stroke="#999"/>')
            parts.append(f'<text x="{left+j*cell+cell/2}" y="{y+cell/2+5}" text-anchor="middle" font-size="22">{value}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if any(OUT_DIR.iterdir()): raise FileExistsError("Use una salida vacía para conservar evidencia")
    records = load_records()
    labels = sorted({record["clase"] for record in records})
    archives, vectors = {}, {}
    try:
        for record in records:
            vectors[record["id_piloto"]] = vector_from_zip(record, archives)
    finally:
        for archive in archives.values():
            archive.close()

    predictions, fold_rows = [], []
    for fold in FOLDS:
        train = [r for r in records if int(r["pliegue_prueba"]) != fold]
        test = [r for r in records if int(r["pliegue_prueba"]) == fold]
        centroids = {label: np.mean([vectors[r["id_piloto"]] for r in train if r["clase"] == label], axis=0)
                     for label in labels}
        correct = 0
        for record in test:
            vector = vectors[record["id_piloto"]]
            distances = {label: float(np.mean((vector - centroid) ** 2)) for label, centroid in centroids.items()}
            ordered = sorted(distances.items(), key=lambda item: item[1])
            predicted, distance = ordered[0]
            confidence = 1 - distance / (ordered[1][1] + 1e-12)
            correct += predicted == record["clase"]
            predictions.append({"id_piloto": record["id_piloto"], "pliegue": fold,
                                "clase_real": record["clase"], "clase_predicha": predicted,
                                "confianza_relativa": round(float(max(0, confidence)), 4)})
        fold_rows.append({"pliegue": fold, "entrenamiento": len(train), "prueba": len(test),
                          "aciertos": correct, "exactitud": round(correct / len(test), 4)})

    truth = [row["clase_real"] for row in predictions]
    predicted = [row["clase_predicha"] for row in predictions]
    matrix = [[sum(t == actual and p == estimate for t, p in zip(truth, predicted))
               for estimate in labels] for actual in labels]
    metrics = class_metrics(labels, truth, predicted)
    accuracy = sum(t == p for t, p in zip(truth, predicted)) / len(truth)
    macro_f1 = sum(row["f1"] for row in metrics.values()) / len(labels)

    write_csv(OUT_DIR / "predicciones_cv5.csv", predictions,
              ["id_piloto", "pliegue", "clase_real", "clase_predicha", "confianza_relativa"])
    write_csv(OUT_DIR / "metricas_por_clase.csv",
              [{"clase": label, **metrics[label]} for label in labels],
              ["clase", "precision", "recall", "f1", "support"])
    write_csv(OUT_DIR / "metricas_por_pliegue.csv", fold_rows,
              ["pliegue", "entrenamiento", "prueba", "aciertos", "exactitud"])
    write_confusion_svg(OUT_DIR / "matriz_confusion_cv5.svg", labels, matrix)
    summary = {"historia": "US-04", "modelo": "Clasificador de centroides RGB 64x64",
               "proposito": "Línea base reproducible, no apta para diagnóstico clínico.",
               "semilla_particion": 3013, "imagenes": len(records), "clases": labels,
               "validacion": "Validación cruzada estratificada de 5 pliegues; cada imagen se predice una vez fuera de entrenamiento.",
               "exactitud_global": round(accuracy, 4), "f1_macro": round(macro_f1, 4),
               "metricas_por_clase": metrics, "matriz_confusion": {"orden": labels, "valores": matrix},
               "metricas_por_pliegue": fold_rows,
               "limitacion": "La similitud visual no equivale a identificación microbiológica; requiere validación clínica y un modelo de mayor capacidad."}
    (OUT_DIR / "resumen_us04.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"exactitud_global": summary["exactitud_global"], "f1_macro": summary["f1_macro"],
                      "imagenes_evaluadas": len(predictions), "salida": str(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
