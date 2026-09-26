import numpy as np, cv2

def hex_rgb(h):
    h=h.strip().lstrip('#'); return np.array([int(h[i:i+2],16) for i in (0,2,4)],np.uint8)

def recolor_rgb(img,editable,target_hex,strength=1.0):
    target=hex_rgb(target_hex); src_lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt_lab=cv2.cvtColor(target.reshape(1,1,3),cv2.COLOR_RGB2LAB)[0,0].astype(np.float32)
    # Preserve source L and use target chroma; mix according to strength.
    out=src_lab.copy(); m=editable.astype(bool)
    out[m,1]=src_lab[m,1]*(1-strength)+tgt_lab[1]*strength
    out[m,2]=src_lab[m,2]*(1-strength)+tgt_lab[2]*strength
    return cv2.cvtColor(np.clip(out,0,255).astype(np.uint8),cv2.COLOR_LAB2RGB)
