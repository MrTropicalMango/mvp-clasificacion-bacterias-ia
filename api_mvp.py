# UNIR · Equipo 3013B. Autoría original conservada en docs/PROCEDENCIA.json.
"""Servicio académico local. Iniciar exclusivamente en 127.0.0.1."""
import json
import os
import secrets
from urllib.parse import urlsplit
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mvp_runtime import load_checkpoint, read_image, infer, MAX_BYTES
from cnn_gradcam_api import LABELS

ROOT = Path(__file__).resolve().parent
DB = Path(os.environ.get('MVP_DB_PATH', ROOT / 'runtime/mvp_bitacora.sqlite'))
WEIGHTS = Path(os.environ.get('MVP_WEIGHTS', ROOT / 'resultados_cnn_cv5/modelo_seleccionado.pt'))
MANIFEST = ROOT / 'configs/weights_sha256.json'


def init():
    DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as connection:
        connection.execute('create table if not exists prediction(id text primary key, payload text not null)')
        connection.execute('create table if not exists revision(id integer primary key, fecha_utc text not null, prediction_id text unique not null, payload text not null)')


@asynccontextmanager
async def lifespan(app):
    init()
    yield


app = FastAPI(title='MVP académico de microbiología', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:8001', 'http://localhost:8001'], allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])


@app.middleware('http')
async def local_only(request: Request, call_next):
    from starlette.responses import JSONResponse
    if request.client and request.client.host not in {'127.0.0.1', '::1', 'testclient'}:
        return JSONResponse({'detail': 'Servicio exclusivamente local'}, status_code=403)
    if urlsplit('//'+request.headers.get('host', '')).hostname not in {'localhost', '127.0.0.1', '::1', 'testserver'}:
        return JSONResponse({'detail': 'Host no autorizado'}, status_code=403)
    origin = request.headers.get('origin')
    if origin and origin not in {'http://127.0.0.1:8001', 'http://localhost:8001'}:
        return JSONResponse({'detail': 'Origen no autorizado'}, status_code=403)
    return await call_next(request)


@app.get('/health')
def health():
    return {'status': 'ok', 'scope': 'academic-local-only', 'weights_available': WEIGHTS.is_file()}


@app.post('/predict')
async def predict(file: UploadFile):
    if file.content_type not in {'image/png', 'image/jpeg'}: raise HTTPException(415, 'Solo PNG/JPEG')
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES: raise HTTPException(413, 'Máximo 10 MB')
    try: image = read_image(data)
    except Exception as error: raise HTTPException(422, 'Imagen inválida o fuera de límites') from error
    if not WEIGHTS.is_file(): raise HTTPException(503, 'No hay pesos verificados')
    try: model, checkpoint = load_checkpoint(WEIGHTS, MANIFEST)
    except (ValueError, KeyError) as error: raise HTTPException(503, 'Pesos incompatibles o no verificados') from error
    result = infer(model, checkpoint, image)
    result['archivo'] = Path(file.filename or 'imagen').name[:120]
    result['prediction_id'] = secrets.token_urlsafe(24)
    stored = {k:v for k,v in result.items() if k != 'gradcam_png_base64'}
    with sqlite3.connect(DB) as connection:
        connection.execute('insert into prediction values(?,?)', (result['prediction_id'], json.dumps(stored)))
    return result


@app.post('/review')
def review(prediction_id: str, accion: str, resultado_final: str = '', nota: str = ''):
    if accion not in {'confirmar', 'corregir', 'no_concluyente'}: raise HTTPException(422, 'Acción inválida')
    if accion == 'corregir' and resultado_final not in LABELS: raise HTTPException(422, 'Clase final inválida')
    if len(nota) > 500: raise HTTPException(422, 'Nota demasiado larga')
    with sqlite3.connect(DB) as connection:
        row = connection.execute('select payload from prediction where id=?', (prediction_id,)).fetchone()
        if not row: raise HTTPException(404, 'Predicción inexistente')
        payload = json.loads(row[0]); payload.update(accion=accion, resultado_final=resultado_final if accion == 'corregir' else (payload['prediccion_presuntiva'] if accion == 'confirmar' else ''), nota=nota)
        try:
            cursor = connection.execute('insert into revision(fecha_utc,prediction_id,payload) values(?,?,?)', (datetime.now(timezone.utc).isoformat(), prediction_id, json.dumps(payload)))
        except sqlite3.IntegrityError as error: raise HTTPException(409, 'Predicción ya revisada') from error
    return {'id': cursor.lastrowid, 'saved': True}


@app.get('/reviews')
def reviews(limit: int = 50, offset: int = 0):
    if not 1 <= limit <= 100 or offset < 0: raise HTTPException(422, 'Paginación inválida')
    with sqlite3.connect(DB) as connection:
        rows = connection.execute('select id,fecha_utc,payload from revision order by id limit ? offset ?', (limit, offset)).fetchall()
    return [dict(id=row[0], fecha_utc=row[1], **json.loads(row[2])) for row in rows]
