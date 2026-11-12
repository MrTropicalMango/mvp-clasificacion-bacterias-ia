"""Entrenamiento reproducible de CNN, selección de modelo y evidencia Grad-CAM.

Uso: python cnn_gradcam_api.py
Entrena dos arquitecturas pequeñas bajo CV-5, guarda métricas, matrices de
confusión, pesos por pliegue y un modelo final para la demostración académica.
No debe usarse para diagnóstico ni decisión terapéutica.
"""
import csv
import os
import platform
import sys
import argparse
import io
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent
MANIFEST = ROOT / "inventario_dataset_piloto/subconjunto_piloto.csv"
RAW_DATA = Path(os.environ.get("MVP_DATA_DIR", ROOT / "dataset_piloto_bruto"))
OUT = Path(os.environ.get("MVP_OUTPUT_DIR", ROOT / "resultados_cnn_cv5"))
LABELS = ["Acinetobacter baumannii", "Actinomyces israelii", "Bacteroides fragilis", "Enterococcus faecalis", "Escherichia coli"]
SEED = 3013
IMAGE_SIZE = 96
TENSOR_CACHE = {}


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class PilotDataset(Dataset):
    def __init__(self, rows): self.rows, self.archives = rows, {}
    def __len__(self): return len(self.rows)
    def image_from_row(self, row):
        if row["archivo_zip"] not in self.archives:
            self.archives[row["archivo_zip"]] = zipfile.ZipFile(RAW_DATA / row["archivo_zip"])
        archive = self.archives[row["archivo_zip"]]
        return Image.open(io.BytesIO(archive.read(row["ruta_en_zip"]))).convert("RGB")
    def __del__(self):
        for archive in self.archives.values(): archive.close()
    def __getitem__(self, index):
        row = self.rows[index]
        key = row["sha256"]
        array = TENSOR_CACHE.get(key)
        if array is None:
            image = self.image_from_row(row).resize((IMAGE_SIZE, IMAGE_SIZE))
            array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
            TENSOR_CACHE[key] = array
        return torch.tensor(array), LABELS.index(row["clase"])


class StandardCNN(nn.Module):
    """CNN convencional pequeña, adecuada al tamaño reducido del piloto."""
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.last_conv = nn.Conv2d(64, 96, 3, padding=1)
        self.head = nn.Sequential(nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.20), nn.Linear(96, 5))
    def forward(self, image): return self.head(self.last_conv(self.block2(self.block1(image))))


class DepthwiseCNN(nn.Module):
    """CNN separable para contrastar una alternativa de menor complejidad."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.depthwise = nn.Sequential(nn.Conv2d(24, 24, 3, padding=1, groups=24), nn.ReLU(), nn.Conv2d(24, 48, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.last_conv = nn.Conv2d(48, 64, 3, padding=1)
        self.head = nn.Sequential(nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.20), nn.Linear(64, 5))
    def forward(self, image): return self.head(self.last_conv(self.depthwise(self.stem(image))))


MODEL_BUILDERS = {"cnn_estandar": StandardCNN, "cnn_depthwise": DepthwiseCNN}


def metrics(truth, prediction):
    confusion = [[sum(actual == row and guessed == column for actual, guessed in zip(truth, prediction)) for column in range(5)] for row in range(5)]
    per_class = []
    for index, label in enumerate(LABELS):
        tp = confusion[index][index]; precision = tp / (sum(confusion[row][index] for row in range(5)) or 1); recall = tp / (sum(confusion[index]) or 1)
        per_class.append({"class": label, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall or 1), "support": sum(confusion[index])})
    return {"accuracy": sum(actual == guessed for actual, guessed in zip(truth, prediction)) / len(truth), "f1_macro": sum(item["f1"] for item in per_class) / 5, "confusion": confusion, "per_class": per_class}


def train_fold(model_name, train_rows, test_rows, epochs, device):
    model = MODEL_BUILDERS[model_name]().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    training = DataLoader(PilotDataset(train_rows), batch_size=48, shuffle=True); model.train(); started = time.perf_counter()
    for _ in range(epochs):
        for image, label in training:
            optimizer.zero_grad(); loss = nn.functional.cross_entropy(model(image.to(device)), label.to(device)); loss.backward(); optimizer.step()
    training_seconds = time.perf_counter() - started; model.eval(); truth, prediction = [], []; test = DataLoader(PilotDataset(test_rows), batch_size=24); started = time.perf_counter()
    with torch.no_grad():
        for image, label in test:
            prediction.extend(model(image.to(device)).argmax(1).cpu().tolist()); truth.extend(label.tolist())
    return model.cpu(), truth, prediction, training_seconds, time.perf_counter() - started


def save_confusion_png(confusion, filename, title):
    margin, cell = 280, 130; canvas = Image.new("RGB", (1100, 1100), "white"); draw = ImageDraw.Draw(canvas); font = ImageFont.load_default(); draw.text((20, 20), title, fill="black", font=font)
    short = [label.split()[0] for label in LABELS]
    for index, label in enumerate(short): draw.text((margin + index * cell + 25, 80), label, fill="black", font=font); draw.text((15, margin + index * cell + 45), label, fill="black", font=font)
    maximum = max(max(row) for row in confusion) or 1
    for row in range(5):
        for column in range(5):
            value = confusion[row][column]; shade = int(245 - 190 * value / maximum); x0, y0 = margin + column * cell, margin + row * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=(shade, shade, 255), outline="black"); draw.text((x0 + 58, y0 + 58), str(value), fill="black", font=font)
    canvas.save(filename)


def save_gradcam(model, row, output_path):
    dataset = PilotDataset([row]); image = dataset.image_from_row(row); array = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32).transpose(2, 0, 1) / 255.0; tensor = torch.tensor(array).unsqueeze(0)
    activations, gradients = [], []
    hook_a = model.last_conv.register_forward_hook(lambda module, inputs, output: activations.append(output)); hook_g = model.last_conv.register_full_backward_hook(lambda module, grad_input, grad_output: gradients.append(grad_output[0]))
    model.eval(); logits = model(tensor); predicted = int(logits.argmax(1).item()); model.zero_grad(); logits[0, predicted].backward(); weights = gradients[0].mean((2, 3), keepdim=True); cam = (weights * activations[0]).sum(1).relu()[0].detach().numpy(); cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heat = Image.fromarray(np.uint8(cam * 255)).resize(image.size); overlay = Image.blend(image, Image.new("RGB", image.size, (255, 0, 0)), 0.42); result = Image.composite(overlay, image, heat.point(lambda value: value * 0.70)).convert("RGB"); result.save(output_path); hook_a.remove(); hook_g.remove()
    return {"file": row["id_piloto"], "expected_class": row["clase"], "predicted_class": LABELS[predicted], "output": output_path.name}


def train_final(model_name, rows, epochs, device):
    model = MODEL_BUILDERS[model_name]().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); loader = DataLoader(PilotDataset(rows), batch_size=48, shuffle=True); model.train()
    for _ in range(epochs):
        for image, label in loader:
            optimizer.zero_grad(); loss = nn.functional.cross_entropy(model(image.to(device)), label.to(device)); loss.backward(); optimizer.step()
    return model.cpu()


def run(epochs=25):
    if epochs < 1: raise ValueError("epochs debe ser positivo")
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()): raise FileExistsError("Use una carpeta de salida vacía para preservar evidencia")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    set_seed(); rows = list(csv.DictReader(MANIFEST.open(encoding="utf8"))); device = "cpu"; report = {"seed": SEED, "images": len(rows), "folds": 5, "epochs": epochs, "device": device, "scope": "Piloto académico; no validación clínica ni uso diagnóstico.", "models": {}}; best = None
    for model_name in MODEL_BUILDERS:
        all_truth, all_prediction, fold_metrics = [], [], []
        for fold in range(1, 6):
            set_seed(SEED + fold); train_rows = [row for row in rows if int(row["pliegue_prueba"]) != fold]; test_rows = [row for row in rows if int(row["pliegue_prueba"]) == fold]
            model, truth, prediction, training_seconds, inference_seconds = train_fold(model_name, train_rows, test_rows, epochs, device); torch.save({"architecture": model_name, "labels": LABELS, "state_dict": model.state_dict(), "image_size": IMAGE_SIZE}, OUT / f"{model_name}_fold{fold}.pt")
            fold_summary = metrics(truth, prediction); fold_summary.update({"fold": fold, "train_images": len(train_rows), "test_images": len(test_rows), "training_seconds": round(training_seconds, 4), "inference_seconds": round(inference_seconds, 4), "inference_ms_per_image": round(1000 * inference_seconds / len(test_rows), 4)}); fold_metrics.append(fold_summary); all_truth.extend(truth); all_prediction.extend(prediction)
        summary = metrics(all_truth, all_prediction); summary["fold_metrics"] = fold_metrics; report["models"][model_name] = summary; save_confusion_png(summary["confusion"], OUT / f"matriz_confusion_{model_name}.png", f"Matriz de confusión: {model_name}")
        if best is None or summary["f1_macro"] > best[1]: best = (model_name, summary["f1_macro"])
    report["environment"] = {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__), "numpy": np.__version__, "threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled()}
    selected_name = best[0]; set_seed(SEED); selected = train_final(selected_name, rows, epochs, device); torch.save({"architecture": selected_name, "labels": LABELS, "state_dict": selected.state_dict(), "image_size": IMAGE_SIZE}, OUT / "modelo_seleccionado.pt")
    report["selected_model"] = selected_name; report["selection_rule"] = "Mayor F1 macro agregada en CV-5 entre las dos CNN evaluadas."; report["gradcam_example"] = save_gradcam(selected, rows[0], OUT / "gradcam_ejemplo.png")
    (OUT / "resumen_cnn.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    with (OUT / "comparacion_modelos.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["modelo", "exactitud", "f1_macro", "tiempo_inferencia_ms_promedio"]); writer.writeheader()
        for name, summary in report["models"].items(): writer.writerow({"modelo": name, "exactitud": round(summary["accuracy"], 6), "f1_macro": round(summary["f1_macro"], 6), "tiempo_inferencia_ms_promedio": round(sum(item["inference_ms_per_image"] for item in summary["fold_metrics"]) / 5, 4)})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    print(json.dumps(run(args.epochs), indent=2))
