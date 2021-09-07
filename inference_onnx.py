import torch
import onnxruntime
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from PIL import Image


# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Inference onnx')
parser.add_argument('--model-onnx', type=str, required=True)
args = parser.parse_args()


# --------------- Main ---------------

src = cv2.imread("test_fg.jpg")
bgr = cv2.imread("test_bg.jpg")

# src = cv2.imread("test_popo_fg.png")
# bgr = cv2.imread("test_popo_bg.png")

# src = cv2.resize(src , (480, 270))
# bgr = cv2.resize(bgr , (480, 270))

# h, w, ch = src.shape
# src = src[0:round(h/2), round(w/2):w, :]
# bgr = bgr[0:round(h/2), round(w/2):w, :]

src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

src = np.expand_dims(np.transpose(src/255.,(2,0,1)),0).astype(np.float32)
bgr = np.expand_dims(np.transpose(bgr/255.,(2,0,1)),0).astype(np.float32)

sess = onnxruntime.InferenceSession(args.model_onnx)
print(sess.get_providers())

# for _ in tqdm(range(10)):
#     out_onnx = sess.run(None, {
#         'src': src,
#         'bgr': bgr
#         })

out_onnx = sess.run(None, {
    'src': src,
    'bgr': bgr
    })

for output_data in zip(out_onnx):
    dst = np.transpose(np.squeeze(output_data[0],0),(1,2,0))
    w,h,ch = dst.shape
    print(w,h,ch)
    if(ch==1):
        pha = dst[:,:,0]
        pha = (pha * 255).astype(np.uint8)
        plt.imshow(pha)
        plt.colorbar
        plt.show()
    else:
        fgr = (dst * 255).astype(np.uint8)      
        plt.imshow(fgr)
        plt.colorbar
        plt.show()   

# show comp image
tgt_bgr = np.zeros((w,h,3), dtype=np.uint8)
tgt_bgr[0:w, 0:h] = [120, 255, 155]
pha = (cv2.cvtColor(pha, cv2.COLOR_GRAY2RGB))
com = (fgr * (pha/255) + tgt_bgr * ((255 - pha)/255)).astype(np.uint8)
plt.imshow(com)
plt.colorbar
plt.show()
