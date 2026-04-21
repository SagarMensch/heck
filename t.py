f = open("log.txt", "w")
f.write("1\n")
import torch
f.write("2\n")
f.write(str(torch.cuda.is_available()))
f.write("\n")
f.write("3\n")
import json
f.write("4\n")
with open("log.txt", "a") as f:
    f.write("5\n")
from transformers import VisionEncoderDecoderModel
f.write("6\n")
m = VisionEncoderDecoderModel.from_pretrained("models/tocr-trained", local_files_only=True)
f.write("7\n")
f.write("done")