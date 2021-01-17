import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import time

model = onnx.load("onnx_mobilenetv2_xga_046875_for_trt.onnx")
engine = backend.prepare(model, device='CUDA:0')

input_src = np.random.random(size=(1,3,768,1024)).astype(np.float32)
input_bgr = np.random.random(size=(1,3,768,1024)).astype(np.float32)

start_time = time.time()
outputs = engine.run([input_src,input_bgr])
end_time = time.time() - start_time
print("infer time=",end_time*1000)

print(len(outputs))

