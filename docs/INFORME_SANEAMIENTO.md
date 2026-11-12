# Saneamiento previo a GitHub — 12 de noviembre de 2026

> Nota de publicación: este informe conserva el diagnóstico previo. El usuario revisó y autorizó el paquete; se publicó el 12 de noviembre de 2026 en https://github.com/MrTropicalMango/mvp-clasificacion-bacterias-ia .

## Dictamen

Copia técnica saneada preparada para revisión. No se creó repositorio, no se publicó contenido y no se reemplazaron originales locales o de Drive. La licencia MIT fue elegida por el usuario, quien confirmó autorización del equipo para publicar y licenciar. La publicación del paquete concreto sigue pendiente de revisión y autorización expresa.

## Base recuperada y versiones

Se recuperó `cnn_gradcam_api.py` del respaldo `auditoria_reproducibilidad/Paquete_Evidencia_Tecnica_sin_imagenes.zip`. Sus 10 822 bytes y SHA-256 `e9e1c7da278686aba5cab7bca0ed9b89fa60018ad1ca9a45e63095c30f140342` coinciden con el manifiesto histórico. Esa versión define las dos arquitecturas, el contrato de checkpoints y el modelo seleccionado que faltaban en el script de Drive de 3 511 bytes. La copia local actual era otra versión y también se conservó intacta.

La recuperación reconcilia interfaces, pero no demuestra que las cifras históricas procedan de una ejecución idéntica en este entorno. Tras configurar CPU, un hilo y determinismo dentro del proceso de entrenamiento, una corrida con PyTorch 2.5.1/Pillow 10.4.0 produjo resultados distintos a los académicos. Se actualizaron las dependencias vulnerables y se generaron dos corridas nuevas, documentadas separadamente.

## Cambios aplicados

- Código compatible entre entrenamiento, API y demos, con carga e inferencia compartidas.
- CPU, semilla e hilos configurados dentro del proceso de entrenamiento. Las salidas deben estar vacías para impedir sobrescrituras accidentales.
- Rutas de datos y salidas configurables; búsqueda recursiva del primer modelo eliminada.
- Carga restringida `weights_only=True`, validación de arquitectura/etiquetas/tamaño y SHA-256 antes de deserializar.
- Límites de tamaño, formato y píxeles antes de decodificar por completo las imágenes.
- API limitada a conexiones, hosts y orígenes locales. Sin exposición automática ni soporte de proxy.
- Revisiones vinculadas a una predicción guardada, con clase/confianza tomadas del servidor, una revisión por predicción y paginación.
- Notebooks sin salidas ni identidades de ejecución. Demos sin Gradio ni enlaces públicos; visualización directa dentro del notebook y página local interactiva.
- Interfaz usa la misma predicción CNN que su Grad-CAM. Incluye arrastrar archivo, revisión, corrección obligatoria y atajos C/R/N con confirmación fuera de campos de entrada. Los registros quedan en SQLite local; no se exporta información automáticamente.
- Evaluación humana bloquea cierre ante códigos repetidos, vacíos o errores estructurales.
- US-08 lee métricas de resultados y verifica pesos; su reporte usa entorno y fecha de ejecución. No presenta métricas constantes como una evaluación nueva.
- Comprobaciones documentales históricas separadas de las pruebas actuales; no certifican por sí mismas una liberación pública.
- MIT y NOTICE separan código, datos, dependencias y pesos. Datos completos, registros humanos, SQLite, resultados binarios y archivos históricos se excluyen mediante lista explícita del ZIP de revisión.

## Verificaciones ejecutadas

1. **Datos:** 295/295 imágenes con hash correcto, 295 huellas únicas, 59 por clase. Los cinco ZIP completos coinciden con el manifiesto histórico. EXIF de las 311 imágenes originales: espacio de color y dimensiones; no campos de texto identificativos detectados. Esto no demuestra independencia por paciente, muestra o microscopio ni descarta duplicados transformados.
2. **Dependencias:** instalación desde cero en Python 3.12/macOS ARM64. `pip check` sin incompatibilidades. Se fijó la resolución completa. Auditoría final sin vulnerabilidades conocidas encontradas para esa resolución en la consulta realizada. No constituye una garantía futura.
3. **Regresiones:** inferencia y Grad-CAM espacial; revisión vinculada; bloqueo de duplicados; archivos inválidos, excesivos y pequeños; origen/host no autorizados; pesos alterados; participantes duplicados; limpieza de notebooks. Resultados en `PRUEBAS_ENTORNO_LIMPIO.txt`.
4. **Navegador:** caso real de carga, predicción, Grad-CAM visible, corrección vacía bloqueada y guardado; sin errores JavaScript. Resultado en `PRUEBA_NAVEGADOR.json`.
5. **Notebooks demo:** ambas celdas ejecutadas con el motor final e imagen pública; inferencia, Grad-CAM y guardado de revisión comprobados. El montaje de Drive y la interfaz de Google Colab no se automatizaron: deben configurarse en el entorno del usuario.
6. **Línea base:** reejecutada sobre las 295 imágenes; exactitud 0.8136 y F1 macro 0.8113.
7. **CNN:** dos ejecuciones completas de ambas arquitecturas, cinco pliegues, 25 épocas y ajuste final. Coincidencia de métricas excluyendo tiempos y de SHA-256 de los once checkpoints entre ambas ejecuciones. Comparación en `COMPARACION_REPRODUCCION.json`.

## Resultados y discrepancia histórica

| Versión | Modelo | Exactitud | F1 macro |
|---|---|---:|---:|
| Académica documentada, no reproducida aquí | CNN estándar | 0.6407 | 0.6418 |
| Académica documentada, no reproducida aquí | CNN depthwise | 0.4644 | 0.4380 |
| Nueva corrida controlada, dependencias antiguas | CNN estándar | 0.7017 | 0.6763 |
| Nueva corrida controlada, dependencias antiguas | CNN depthwise | 0.4271 | 0.4187 |
| Nueva versión saneada; dos corridas idénticas | CNN estándar | 0.7254 | 0.7052 |
| Nueva versión saneada; dos corridas idénticas | CNN depthwise | 0.4203 | 0.4126 |
| Línea base reejecutada | Centroides RGB | 0.8136 | 0.8113 |

La línea base sigue superando a ambas CNN. La CNN estándar se conserva para la demostración de Grad-CAM. Los pesos por defecto de la copia saneada proceden de la primera corrida actualizada, corroborada por la segunda; sus hashes están en `configs/weights_sha256.json`. No se cambiaron las métricas de la tesis ni se sobrescribieron evidencias históricas.

No se atribuye la discrepancia histórica a una causa única sin evidencia suficiente. Configuración efectiva del proceso, versiones y operaciones numéricas son factores a investigar. Un portafolio público debe distinguir explícitamente los resultados académicos de los nuevos; no debe afirmar reproducción exacta de los primeros.

## Dependencias y decisión sobre Gradio

El conjunto final usa PyTorch 2.13.0, Pillow 13.0.0, FastAPI 0.162.5 y python-multipart 0.0.31, con NumPy 1.26.4, Uvicorn 0.34.0 y HTTPX 0.28.1. El archivo de bloqueo incluye las transitivas.

La revisión completa detectó vulnerabilidades también en las versiones antiguas de PyTorch, Pillow y python-multipart. Gradio 5.44.1 tenía avisos adicionales. Incluso la versión consultada más reciente de Gradio exigía Pillow <13, incompatible con la corrección necesaria. Se retiró Gradio del candidato y se conservaron demos de notebook e interfaz HTML local. Los informes anteriores de dependencias se conservan internamente como diagnóstico y no describen el estado final.

## Límites y revisión antes de publicar

- Revisión editorial del README, MIT/NOTICE y presentación del proyecto personal.
- Aceptar la separación de resultados históricos y nuevos. Si se desea afirmar reproducción histórica exacta, investigar y resolver primero la discrepancia.
- La versión se probó en macOS ARM64/Python 3.12; otros sistemas y Colab requieren validación específica.
- Los ZIP y pesos no forman parte del paquete público propuesto; los comandos de inferencia e integración requieren restaurarlos según el manifiesto. No se propone publicar registros humanos ni información del equipo distinta de la autoría autorizada.
- Sin validación clínica, calibración, prueba externa o controles de servicio multiusuario. Grad-CAM no acredita causalidad.
- Escaneo de patrones de secretos y exclusión por lista explícita reducen exposición; no sustituyen una evaluación de seguridad de producción.

El paquete se entrega para revisión. Crear el repositorio y hacerlo público son pasos posteriores que no se ejecutaron.
