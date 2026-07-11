import numpy as np
import os
import glob

def check_handedness_balance(signer_dir):
    lh_active, rh_active, total = 0, 0, 0
    for npy_path in glob.glob(os.path.join(signer_dir, "**/*.npy"), recursive=True):
        seq = np.load(npy_path)  # (frames, 126)
        lh = seq[:, :63]
        rh = seq[:, 63:]
        lh_active += np.any(lh != 0, axis=1).sum()
        rh_active += np.any(rh != 0, axis=1).sum()
        total += seq.shape[0]
    print(f"  Left hand active:  {lh_active/total*100:.1f}% of frames")
    print(f"  Right hand active: {rh_active/total*100:.1f}% of frames")

for signer in ["01", "02", "03"]:
    print(f"Signer {signer}:")
    check_handedness_balance(f"output/npy/train/{signer}")
