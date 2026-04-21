print("START", file=open("s1.txt", "w"))
import torch
print(f"CUDA={torch.cuda.is_available()}", file=open("s1.txt", "a"))
print("DONE", file=open("s1.txt", "a"))