# MVP académico de clasificación presuntiva de bacterias

Proyecto de titulación del Equipo 3013B, UNIR. Código preparado para revisión del portafolio personal de Octavio Mejía. Licencia MIT para el código; datos con atribución separada en NOTICE.md.

**Estado: candidato interno saneado; publicación pendiente de revisión y autorización expresa.**

El prototipo compara centroides RGB y dos CNN pequeñas, muestra Grad-CAM y permite revisión humana. No es diagnóstico, no trabaja con datos de pacientes ni acredita validación clínica. La confianza es relativa al modelo y no una probabilidad clínica calibrada.

## Organización

- `cnn_gradcam_api.py`: arquitecturas y entrenamiento CPU con CV-5.
- `mvp_runtime.py`: carga restringida, validación y Grad-CAM compartidos.
- `api_mvp.py`: inferencia y revisión SQLite con identificador de predicción.
- `demo_mvp.py`, `notebooks/`: recorridos sin rutas personales ni servidores públicos.
- `prototipo_mvp/index.html`: interfaz de la API; arrastrar imágenes, revisión y atajos C/R/N con confirmación.
- `tests/`: regresiones verificables. Las antiguas comprobaciones documentales se conservan fuera del paquete público, como referencia interna.
- `configs/weights_sha256.json`: huellas aprobadas para los pesos de referencia. Una corrida nueva necesita revisión de sus pesos y actualización explícita de este manifiesto.
- `inventario_dataset_piloto/`: índices de datos públicos y particiones; los ZIP no se redistribuyen.
- `docs/`: cambios, evidencia de verificación, procedencia y límites.

## Instalación

Python 3.12. La plataforma comprobada es macOS ARM64. Para reproducir exactamente el entorno resuelto en esa plataforma:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock-macos-arm64-py312.txt
python -m pip check
python tests/test_saneamiento.py
```

Las pruebas integradas necesitan restaurar los pesos de referencia descritos en la siguiente sección; el ZIP de revisión no los contiene.

`requirements_mvp_cnn.txt` expresa las dependencias directas; el archivo de bloqueo recoge la resolución comprobada. No se garantiza identidad binaria de resultados entre sistemas o versiones. La actualización de PyTorch/Pillow se registra como una nueva versión experimental, sin reemplazar métricas académicas históricas.

## Datos y pesos

Descargue la versión 1 de la fuente enlazada en NOTICE.md y prepare los ZIP con los nombres de `inventario_dataset_piloto/subconjunto_piloto.csv`. Configure `MVP_DATA_DIR` con su carpeta. Cada imagen debe coincidir con el SHA-256 del manifiesto; los 295 registros corresponden a 59 imágenes por clase. Dos ZIP del respaldo histórico se dividieron en partes; deben concatenarse y comprobarse contra su manifiesto original.

El paquete de revisión contiene código, no ZIP, imágenes, pesos ni SQLite. La copia interna de trabajo conserva pesos de referencia con sus hashes. Para inferencia local necesita disponer del modelo indicado por `configs/weights_sha256.json` en `resultados_cnn_cv5/`. No cargue modelos de origen desconocido. No edite el manifiesto para eludir una discrepancia: revise primero procedencia, arquitectura, etiquetas y resultados.

## Interfaz local

Desde esta carpeta, con el entorno activo:

```sh
uvicorn api_mvp:app --host 127.0.0.1 --port 8000
python -m http.server 8001 --bind 127.0.0.1 --directory prototipo_mvp
```

Abra http://127.0.0.1:8001. Use una imagen PNG/JPEG pública, mínimo 224 × 224, máximo 16 millones de píxeles y 10 MB. La API guarda únicamente metadatos de predicción/revisión en `runtime/`, sin persistir la imagen. Los registros y pesos quedan excluidos del paquete público.

Las revisiones toman clase y confianza del registro de inferencia, exigen un identificador válido y admiten una sola revisión por predicción. La API es exclusivamente local y no debe ponerse detrás de un proxy ni exponerse a Internet. No es un servicio multiusuario ni incorpora gestión de identidades.

## Entrenamiento y reproducción

Use dos carpetas de salida nuevas y vacías. El entrenamiento fija CPU, un hilo, semilla 3013 y algoritmos deterministas dentro de su propio proceso:

```sh
export MVP_DATA_DIR=/ruta/a/dataset_piloto_bruto
MVP_OUTPUT_DIR=/ruta/a/corrida_a python cnn_gradcam_api.py --epochs 25
MVP_OUTPUT_DIR=/ruta/a/corrida_b python cnn_gradcam_api.py --epochs 25
python verificar_reproduccion.py /ruta/a/corrida_a /ruta/a/corrida_b
```

Se comparan métricas sin tiempos y los once checkpoints. El comparador requiere 25 épocas, 295 imágenes y cinco pliegues de ambas arquitecturas. `ejecutar_us08.py` mide inferencia en memoria, excluyendo lectura y preprocesamiento; lee métricas previas y no las recalcula a partir de nuevas predicciones. No confundirlo con reentrenamiento.

## Notebooks

Instale las dependencias en el kernel y configure `MVP_PROJECT_ROOT`, `MVP_DATA_DIR` y una salida nueva `MVP_OUTPUT_DIR`. Las demos requieren `MVP_EXAMPLE_IMAGE`; la comparación requiere `MVP_RUN_A` y `MVP_RUN_B`. En Colab monte Drive manualmente y seleccione la carpeta exacta. No se busca el primer modelo de toda la unidad. Las demos muestran resultados dentro del notebook y permiten registrar revisión con una llamada explícita; la interfaz interactiva completa se ejecuta localmente en el navegador.

## Alcance científico

Una fuente pública, cinco clases y 295 imágenes no permiten inferir generalización clínica. SHA-256 descarta duplicados binarios, no imágenes visualmente relacionadas ni dependencia por muestra o microscopio. No hay prueba externa ni calibración clínica. Grad-CAM no explica causalidad. La comparación entre arquitecturas usa los mismos pliegues para seleccionar la CNN; no es una estimación externa del modelo seleccionado.
