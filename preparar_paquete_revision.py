"""Empaqueta una lista explícita de archivos para revisión, sin publicar."""
from pathlib import Path
import hashlib,json,re,zipfile
ROOT=Path(__file__).resolve().parent
FILES=['README.md','LICENSE','NOTICE.md','.gitignore','requirements_mvp_cnn.txt','requirements-demo.txt','requirements-lock-macos-arm64-py312.txt','cnn_gradcam_api.py','mvp_runtime.py','api_mvp.py','demo_mvp.py','entrenar_linea_base_us04.py','ejecutar_us08.py','verificar_reproduccion.py','preparar_paquete_revision.py','prototipo_mvp/index.html','prototipo_mvp/evaluar_validacion_us11_us12.py','tests/test_saneamiento.py','configs/weights_sha256.json','docs/PROCEDENCIA.json','docs/MATRIZ_DERECHOS.md','docs/INFORME_SANEAMIENTO.md','docs/VALIDACION_DATOS.json','docs/AUDITORIA_DEPENDENCIAS_FINAL.json','docs/PRUEBAS_ENTORNO_LIMPIO.txt','docs/PRUEBA_NAVEGADOR.json','docs/COMPARACION_REPRODUCCION.json','docs/RESULTADOS_NUEVOS.json']
FILES += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'notebooks').glob('*.ipynb'))]
FILES += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'inventario_dataset_piloto').glob('*.csv'))]
PATTERNS=[r'AKIA[0-9A-Z]{16}',r'gh[pousr]_[A-Za-z0-9]{20,}',r'AIza[0-9A-Za-z_-]{30,}',r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----']
def build():
    manifest={};findings=[]
    for name in FILES:
        data=(ROOT/name).read_bytes();text=data.decode('utf-8')
        if name!='preparar_paquete_revision.py' and any(re.search(pattern,text) for pattern in PATTERNS):findings.append(name)
        if 'userId' in text and name.endswith('.ipynb'):findings.append(name)
        manifest[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    if findings:raise ValueError('Revisar posibles secretos: '+str(findings))
    out=ROOT/'paquete_revision_codigo.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:archive.write(ROOT/name,name)
        archive.writestr('MANIFIESTO_CODIGO.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(out) as archive:
        assert set(archive.namelist())==set(FILES)|{'MANIFIESTO_CODIGO.json'}
        for name,entry in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==entry['sha256']
    report={'files':len(FILES),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size,'secret_pattern_matches':[],'excluded':['datasets ZIP','imágenes','pesos','SQLite','históricos','registros humanos','entornos','resultados experimentales binarios'],'publication':'NO PUBLICADO; revisión y autorización pendientes'}
    (ROOT/'docs/PAQUETE_REVISION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':build()
