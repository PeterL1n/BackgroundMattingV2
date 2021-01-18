import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import time
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt

model = onnx.load("onnx_mobilenetv2_hd_for_trt.onnx")
engine = backend.prepare(model, device='CUDA:0')

input_src = np.random.random(size=(1,3,1080,1920)).astype(np.float32)
input_bgr = np.random.random(size=(1,3,1080,1920)).astype(np.float32)

for _ in tqdm(range(100)):
	#start_time = time.time()
	outputs = engine.run([input_src,input_bgr])[:2]
	#end_time = time.time() - start_time
	#print("infer time=",end_time*1000)
	#print(len(outputs))


src = cv2.imread("test_img_fg.png")
bgr = cv2.imread("test_img_bg.png")

src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

src = np.expand_dims(np.transpose(src/255.,(2,0,1)),0).astype(np.float32)
bgr = np.expand_dims(np.transpose(bgr/255.,(2,0,1)),0).astype(np.float32)

src = np.array(src, dtype=src.dtype, order='C')
bgr = np.array(bgr, dtype=src.dtype, order='C')

outputs = engine.run([src, bgr])

for i in range(len(outputs)):
	print("data=",i)
	res = np.transpose(np.squeeze(outputs[i],0),(1,2,0))
	plt.imshow(res)
	plt.colorbar
	plt.show()


