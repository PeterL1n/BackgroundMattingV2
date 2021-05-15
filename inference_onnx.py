import torch
import onnxruntime
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

onnx_model = "test_output.onnx"
#onnx_model = "test_output_simple.onnx"


src = cv2.imread("test_fg.jpg")
bgr = cv2.imread("test_bg.jpg")

#src = cv2.resize(src , (480, 270))
#bgr = cv2.resize(bgr , (480, 270))

#h, w, ch = src.shape
#src = src[0:round(h/2), round(w/2):w, :]
#bgr = bgr[0:round(h/2), round(w/2):w, :]

src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

src = np.expand_dims(np.transpose(src/255.,(2,0,1)),0).astype(np.float32)
bgr = np.expand_dims(np.transpose(bgr/255.,(2,0,1)),0).astype(np.float32)

sess = onnxruntime.InferenceSession(onnx_model)
print(sess.get_providers())

for _ in tqdm(range(30)):
    out_onnx = sess.run(None, {
        'src': src,
        'bgr': bgr
        })

out_onnx = sess.run(None, {
    'src': src,
    'bgr': bgr
    })

for output_data in zip(out_onnx):
    res = np.transpose(np.squeeze(output_data[0],0),(1,2,0))
    print(res.shape)
    cv2.imwrite('output.jpg', res*255)
    plt.imshow(res)
    plt.colorbar
    plt.show()


