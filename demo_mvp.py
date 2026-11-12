"""Demostración compartida sin servidor público ni dependencias de interfaz."""
import base64,csv,io
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image
from mvp_runtime import load_checkpoint,read_image,infer
from cnn_gradcam_api import LABELS

def run_demo(root,image_path):
    root=Path(root)
    model,checkpoint=load_checkpoint(root/'resultados_cnn_cv5/modelo_seleccionado.pt',root/'configs/weights_sha256.json')
    with Path(image_path).open('rb') as source: data=source.read(10*1024*1024+1)
    result=infer(model,checkpoint,read_image(data))
    heat=Image.open(io.BytesIO(base64.b64decode(result.pop('gradcam_png_base64'))))
    return result,heat

def save_review(root,result,action,correction=''):
    if action not in {'confirmar','corregir','no_concluyente'}: raise ValueError('Acción inválida')
    if action=='corregir' and correction not in LABELS: raise ValueError('Seleccione clase final')
    row={'fecha_utc':datetime.now(timezone.utc).isoformat(),'modelo':result['modelo'],'prediccion':result['prediccion_presuntiva'],'accion':action,'resultado_final':correction if action=='corregir' else (result['prediccion_presuntiva'] if action=='confirmar' else '')}
    path=Path(root)/'runtime/demo_reviews.csv';path.parent.mkdir(exist_ok=True);exists=path.exists()
    with path.open('a',newline='',encoding='utf-8') as target:
        writer=csv.DictWriter(target,fieldnames=list(row))
        if not exists:writer.writeheader()
        writer.writerow(row)
    return path
