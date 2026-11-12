"""Compara métricas y pesos de corridas completas, omitiendo tiempos."""
from pathlib import Path
import hashlib,json,sys
TIMES={'training_seconds','inference_seconds','inference_ms_per_image'}
def clean(x):
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items() if k not in TIMES|{'environment'}}
    if isinstance(x,list): return [clean(v) for v in x]
    return x

def compare(a,b):
    a,b=Path(a),Path(b)
    x=json.loads((a/'resumen_cnn.json').read_text()); y=json.loads((b/'resumen_cnn.json').read_text())
    if not (x['epochs']==y['epochs']==25 and x['images']==y['images']==295): raise ValueError('Corridas incompletas o parámetros diferentes')
    if len(x['models'])!=2 or len(y['models'])!=2: raise ValueError('Se requieren dos arquitecturas')
    for run in (x,y):
        if not all(len(m['fold_metrics'])==5 for m in run['models'].values()): raise ValueError('Se requieren cinco pliegues por arquitectura')
    checks={'metrics_equal':clean(x)==clean(y)}
    weights=sorted(p.name for p in a.glob('*.pt'))
    checks['weights_inventory_equal']=weights==sorted(p.name for p in b.glob('*.pt')) and len(weights)==11
    checks['weights_equal']=all(hashlib.sha256((a/n).read_bytes()).digest()==hashlib.sha256((b/n).read_bytes()).digest() for n in weights)
    return checks
if __name__=='__main__':
    result=compare(*sys.argv[1:]);print(json.dumps(result,indent=2));raise SystemExit(0 if all(result.values()) else 1)
