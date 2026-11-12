import csv
import importlib.util
import io
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from PIL import Image
import api_mvp as api
from mvp_runtime import load_checkpoint

spec=importlib.util.spec_from_file_location('human',ROOT/'prototipo_mvp/evaluar_validacion_us11_us12.py');human=importlib.util.module_from_spec(spec);spec.loader.exec_module(human)

class Regression(unittest.TestCase):
 def test_duplicate_human_records_block_close(self):
  with TemporaryDirectory() as tmp:
   path=Path(tmp)/'records.csv';fields=['codigo',*human.CAMPOS_SI_NO,'accion_accidental']
   with path.open('w',newline='') as out:
    writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
    for _ in range(5):writer.writerow(dict(codigo='P1',accion_accidental='no',**{key:'si' for key in human.CAMPOS_SI_NO}))
   self.assertEqual(human.evaluar(path)['estado_general'],'PENDIENTE')
 def test_unique_human_records(self):
  with TemporaryDirectory() as tmp:
   p=Path(tmp)/'r.csv';fields=['codigo',*human.CAMPOS_SI_NO,'accion_accidental']
   with p.open('w',newline='') as out:
    writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
    for i in range(5):writer.writerow(dict(codigo=f'P{i}',accion_accidental='no',**{key:'si' for key in human.CAMPOS_SI_NO}))
   self.assertEqual(human.evaluar(p)['estado_general'],'LISTO_PARA_CIERRE')
 def test_api_inference_and_linked_review(self):
  with TemporaryDirectory() as tmp:
   api.DB=Path(tmp)/'test.sqlite'
   with TestClient(api.app) as client:
    self.assertEqual(client.get('/health').status_code,200)
    self.assertEqual(client.get('/health',headers={'host':'attacker.example'}).status_code,403)
    self.assertEqual(client.post('/predict',files={'file':('large.png',b'0'*(10*1024*1024+1),'image/png')}).status_code,413)
    self.assertEqual(client.post('/predict',files={'file':('x.txt',b'plain','text/plain')}).status_code,415)
    self.assertEqual(client.get('/health',headers={'origin':'https://example.com'}).status_code,403)
    self.assertEqual(client.post('/review',params={'prediction_id':'invented','accion':'confirmar'}).status_code,404)
    image=Image.new('RGB',(224,224),'white');buffer=io.BytesIO();image.save(buffer,format='PNG')
    response=client.post('/predict',files={'file':('synthetic.png',buffer.getvalue(),'image/png')});self.assertEqual(response.status_code,200,response.text)
    data=response.json();self.assertTrue(data['review_required']);self.assertGreater(min(data['gradcam_shape']),1)
    params={'prediction_id':data['prediction_id'],'accion':'confirmar'}
    self.assertEqual(client.post('/review',params=params).status_code,200)
    self.assertEqual(client.post('/review',params=params).status_code,409)
    self.assertEqual(client.get('/reviews').json()[0]['prediccion_presuntiva'],data['prediccion_presuntiva'])
    self.assertEqual(client.post('/predict',files={'file':('bad.png',b'bad','image/png')}).status_code,422)
    buffer=io.BytesIO();Image.new('RGB',(10,10)).save(buffer,format='PNG')
    self.assertEqual(client.post('/predict',files={'file':('small.png',buffer.getvalue(),'image/png')}).status_code,422)
 def test_hash_tampering_is_rejected(self):
  with TemporaryDirectory() as tmp:
   p=Path(tmp)/'modelo_seleccionado.pt';p.write_bytes(b'changed')
   with self.assertRaises(ValueError):load_checkpoint(p,ROOT/'configs/weights_sha256.json')
 def test_notebooks_are_clean_and_private(self):
  for p in (ROOT/'notebooks').glob('*.ipynb'):
   text=p.read_text();n=json.loads(text)
   self.assertNotIn('share=True',text);self.assertNotIn('userId',text);self.assertNotIn('/content/drive/MyDrive',text)
   for c in n['cells']:
    if c['cell_type']=='code':self.assertEqual(c['outputs'],[])
if __name__=='__main__':unittest.main(verbosity=2)
