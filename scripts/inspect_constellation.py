import numpy as np

a = np.load("data/constellation.npy", allow_pickle=True)
print("dtype       :", a.dtype)
print("shape       :", a.shape)
print("size        :", a.size)
print("first 8     :", a.ravel()[:8])
print()
flat = a.ravel().astype(np.complex128)
print("|z| min/max :", float(np.min(np.abs(flat))), float(np.max(np.abs(flat))))
print("|z| mean    :", float(np.mean(np.abs(flat))))
print("|z| stdev   :", float(np.std(np.abs(flat))))
print()
print("histogram of |z| (10 bins):")
hist, edges = np.histogram(np.abs(flat), bins=10)
for h, e in zip(hist, edges):
    print(f"  >= {e:8.2f}: {h}")
print()
print("phase histogram (16 bins, π-units):")
phase = np.angle(flat) / np.pi
ph_hist, ph_edges = np.histogram(phase, bins=16, range=(-1, 1))
for h, e in zip(ph_hist, ph_edges):
    print(f"  >= {e:+.3f}π: {h}")
