"""Quick GPU diagnostic: time cpsam on 3 small images."""
import time
import numpy as np
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from cellpose import models

gpu = torch.cuda.is_available()
print(f"Loading cpsam model (gpu={gpu})...")
t0 = time.time()
model = models.CellposeModel(gpu=gpu, pretrained_model="cpsam")
print(f"Model loaded in {time.time()-t0:.1f}s")

imgs = [np.random.rand(123, 155).astype(np.float32) for _ in range(3)]
print("Running eval on 3 images (batch)...")
t1 = time.time()
masks, _, _ = model.eval(imgs, diameter=70, channels=None, normalize=False)
elapsed = time.time() - t1
print(f"3 images done in {elapsed:.1f}s = {elapsed/3:.2f}s per image")

# Also time cyto3 for comparison
print("\nLoading cyto3 model...")
t2 = time.time()
model3 = models.CellposeModel(gpu=gpu, pretrained_model="cyto3")
print(f"cyto3 loaded in {time.time()-t2:.1f}s")
t3 = time.time()
masks3, _, _ = model3.eval(imgs, diameter=70, channels=None, normalize=False)
elapsed3 = time.time() - t3
print(f"cyto3: 3 images in {elapsed3:.1f}s = {elapsed3/3:.2f}s per image")
print("Done!")
