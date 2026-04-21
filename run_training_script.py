open("training_output.txt", "w").write("START\n")

from transformers import VisionEncoderDecoderModel
open("training_output.txt", "a").write("1\n")
try:
    cache = r"C:\Users\aigcp_gpuadmin\.cache\huggingface\hub\models--microsoft--trocr-base-handwritten\snapshots\eaacaf452b06415df8f10bb6fad3a4c11e609406"
    model = VisionEncoderDecoderModel.from_pretrained(cache)
    open("training_output.txt", "a").write("MODEL OK\n")
except Exception as e:
    open("training_output.txt", "a").write(f"ERROR: {e}\n")

open("training_output.txt", "a").write("DONE\n")