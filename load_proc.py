import sys
sys.stderr = open('err.log', 'w')
print('1', flush=True)
from transformers import TrOCRProcessor
print('2', flush=True)
cache = r'C:\Users\aigcp_gpuadmin\.cache\huggingface\hub\models--microsoft--trocr-base-handwritten\snapshots\eaacaf452b06415df8f10bb6fad3a4c11e609406'
print('3', flush=True)
p = TrOCRProcessor.from_pretrained(cache)
print('4', flush=True)