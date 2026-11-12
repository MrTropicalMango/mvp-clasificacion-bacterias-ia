"""Carga restringida e inferencia compartida del MVP académico."""
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from cnn_gradcam_api import MODEL_BUILDERS, LABELS, IMAGE_SIZE

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000


def load_checkpoint(path, manifest):
    path = Path(path)
    expected = json.loads(Path(manifest).read_text())[path.name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Hash de pesos diferente al manifiesto aprobado')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') not in MODEL_BUILDERS or checkpoint.get('labels') != LABELS or checkpoint.get('image_size') != IMAGE_SIZE:
        raise ValueError('Contrato de checkpoint incompatible')
    model = MODEL_BUILDERS[checkpoint['architecture']]()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    return model, checkpoint


def read_image(data):
    if len(data) > MAX_BYTES: raise ValueError('Máximo 10 MB')
    with Image.open(io.BytesIO(data)) as source:
        if source.format not in {'PNG', 'JPEG'}: raise ValueError('Solo PNG/JPEG')
        if source.width < 224 or source.height < 224: raise ValueError('Mínimo 224 × 224 píxeles')
        if source.width * source.height > MAX_PIXELS: raise ValueError('Máximo 16 millones de píxeles')
        image = source.convert('RGB')
    return image


def infer(model, checkpoint, image):
    array = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.tensor(array).unsqueeze(0)
    activations, gradients = [], []
    ha = model.last_conv.register_forward_hook(lambda m, i, o: activations.append(o))
    hg = model.last_conv.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))
    try:
        model.zero_grad()
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted = int(probabilities.argmax().item())
        logits[0, predicted].backward()
        cam = (gradients[0].mean((2, 3), keepdim=True) * activations[0]).sum(1).relu()[0].detach().numpy()
        if min(cam.shape) <= 1: raise ValueError('Grad-CAM carece de resolución espacial')
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        mask = Image.fromarray(np.uint8(cam * 255)).resize(image.size).point(lambda v: v * .70)
        overlay = Image.composite(Image.blend(image, Image.new('RGB', image.size, (255, 0, 0)), .42), image, mask)
        buffer = io.BytesIO(); overlay.save(buffer, format='PNG')
        return {'modelo': checkpoint['architecture'], 'prediccion_presuntiva': LABELS[predicted], 'confianza': round(float(probabilities[predicted].detach()), 6), 'review_required': True, 'gradcam_png_base64': base64.b64encode(buffer.getvalue()).decode(), 'gradcam_shape': list(cam.shape), 'limit': 'Piloto académico; no constituye diagnóstico.'}
    finally:
        ha.remove(); hg.remove(); model.zero_grad()
