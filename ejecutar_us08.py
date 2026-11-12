# UNIR
# Seminario de Innovación en Inteligencia Artificial
# Equipo 3013B
# Integrantes: Durán García Sergio Israel; Mallorquín Cabral Christian Javier; Mejía Sandoval Octavio Francisco; Velásquez Irías Fausto Rafael

"""US-08: comparación reproducible de calidad y latencia del MVP.

La calidad se toma de las corridas científicas auditadas. La latencia se
caracteriza en este equipo con CPU, un hilo, lote 1, calentamiento previo y
295 imágenes. Los tiempos no son reproducibles bit a bit entre equipos.
"""
from __future__ import annotations

import csv
import os
import hashlib
import io
import json
import platform
import statistics
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from cnn_gradcam_api import LABELS, MODEL_BUILDERS
from mvp_runtime import load_checkpoint


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "inventario_dataset_piloto" / "subconjunto_piloto.csv"
RAW = Path(os.environ.get("MVP_DATA_DIR", ROOT / "dataset_piloto_bruto"))
CNN = Path(os.environ.get("MVP_RESULTS_DIR", ROOT / "resultados_cnn_cv5"))
OUT = ROOT / "evidencia_us08"
SEED = 3013


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_images(rows):
    archives = {}
    images64, images96 = [], []
    try:
        for row in rows:
            if row["archivo_zip"] not in archives:
                archives[row["archivo_zip"]] = zipfile.ZipFile(RAW / row["archivo_zip"])
            archive = archives[row["archivo_zip"]]
            image = Image.open(io.BytesIO(archive.read(row["ruta_en_zip"]))).convert("RGB")
            a64 = np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1) / 255.0
            a96 = np.asarray(image.resize((96, 96), Image.Resampling.BILINEAR), dtype=np.float32).transpose(2, 0, 1) / 255.0
            images64.append(a64)
            images96.append(a96)
    finally:
        for archive in archives.values():
            archive.close()
    return np.stack(images64), torch.from_numpy(np.stack(images96))


def baseline_centroids(rows, vectors):
    centroids = {}
    for label in LABELS:
        idx = [i for i, row in enumerate(rows) if row["clase"] == label]
        centroids[label] = vectors[idx].mean(axis=0)
    return np.stack([centroids[label] for label in LABELS])


def measure_ms(operation, warmup=2, repetitions=10):
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return samples


def main():
    OUT.mkdir(exist_ok=True)
    if any(OUT.iterdir()): raise FileExistsError("Use una copia sin resultados US-08 previos")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    vectors64, tensors96 = load_images(rows)
    centroids = baseline_centroids(rows, vectors64)

    def infer_baseline():
        for vector in vectors64:
            np.mean((centroids - vector) ** 2, axis=1).argmin()

    latency = {}
    baseline_samples = measure_ms(infer_baseline)
    latency["centroides_rgb"] = [value / len(rows) for value in baseline_samples]

    with torch.inference_mode():
        for name in ("cnn_estandar", "cnn_depthwise"):
            model, checkpoint = load_checkpoint(CNN / f"{name}_fold1.pt", ROOT / "configs/weights_sha256.json")

            def infer_cnn(model=model):
                for tensor in tensors96:
                    model(tensor.unsqueeze(0)).argmax(1)

            samples = measure_ms(infer_cnn)
            latency[name] = [value / len(rows) for value in samples]

    baseline = json.loads((ROOT / "resultados_us04_linea_base/resumen_us04.json").read_text())
    cnn = json.loads((CNN / "resumen_cnn.json").read_text())
    quality = {name: {"accuracy": cnn["models"][name]["accuracy"], "f1_macro": cnn["models"][name]["f1_macro"], "source": "resumen_cnn.json: resultados previos, no reentrenados por US-08"} for name in MODEL_BUILDERS}
    quality["centroides_rgb"] = {"accuracy": baseline["exactitud_global"], "f1_macro": baseline["f1_macro"], "source": "resumen_us04.json: resultados previos"}
    summary = {
        "historia": "US-08",
        "fecha_ejecucion": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocolo": {"dispositivo": "cpu", "hilos_torch": 1, "batch_size": 1, "warmup": 2, "repeticiones": 10, "imagenes_por_repeticion": len(rows), "semilla": SEED},
        "entorno": {"python": platform.python_version(), "sistema": platform.platform(), "numpy": np.__version__, "pillow": Image.__version__, "torch": torch.__version__},
        "resultados": {},
        "decision_lean": "PERSEVERAR de forma acotada: conservar la línea base como referencia de calidad y la CNN estándar como soporte de Grad-CAM; no ampliar a uso clínico hasta contar con validación externa, calibración y evaluación primaria de usuarios.",
        "limitaciones": ["La latencia caracteriza este equipo y puede variar en otro hardware.", "Las métricas corresponden al piloto de 295 imágenes y no prueban eficacia clínica.", "La CNN estándar se selecciona entre CNN por F1 macro, pero no supera a la línea base."],
    }
    for name, samples in latency.items():
        summary["resultados"][name] = {**quality[name], "latencia_ms_por_imagen_media": statistics.mean(samples), "latencia_ms_por_imagen_mediana": statistics.median(samples), "latencia_ms_por_imagen_min": min(samples), "latencia_ms_por_imagen_max": max(samples), "muestras_ms_por_imagen": samples}
    (OUT / "resultado_us08.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "comparacion_us08.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["modelo", "exactitud_cv5", "f1_macro", "latencia_ms_imagen_media", "latencia_ms_imagen_mediana", "decision"])
        for name, data in summary["resultados"].items():
            decision = "referencia de calidad" if name == "centroides_rgb" else ("soporte Grad-CAM" if name == "cnn_estandar" else "descartada")
            writer.writerow([name, f'{data["accuracy"]:.6f}', f'{data["f1_macro"]:.6f}', f'{data["latencia_ms_por_imagen_media"]:.4f}', f'{data["latencia_ms_por_imagen_mediana"]:.4f}', decision])
    standard_ms = summary["resultados"]["cnn_estandar"]["latencia_ms_por_imagen_media"]
    depthwise_ms = summary["resultados"]["cnn_depthwise"]["latencia_ms_por_imagen_media"]
    faster_pct = 100 * (standard_ms - depthwise_ms) / standard_ms
    report = "# Medición US-08\n\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n\nLatencia medida en memoria; excluye lectura y preprocesamiento. Calidad leída de resultados previos. No demuestra eficacia clínica.\n"
    (OUT / "REPORTE_US08.md").write_text(report, encoding="utf-8")
    manifest_files = [Path("ejecutar_us08.py"), Path("evidencia_us08/REPORTE_US08.md"), Path("evidencia_us08/resultado_us08.json"), Path("evidencia_us08/comparacion_us08.csv"), Path("resultados_us04_linea_base/resumen_us04.json")]
    with (OUT / "MANIFIESTO_SHA256_US08.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["ruta_relativa", "bytes", "sha256"])
        for relative in manifest_files:
            path = ROOT / relative; writer.writerow([relative.as_posix(), path.stat().st_size, sha256(path)])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
