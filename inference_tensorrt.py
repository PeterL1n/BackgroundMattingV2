import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np

trt_engine_path = 'onnx_mobilenetv2_hd_fortrt.trt.trt'
# initialize
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
runtime = trt.Runtime(TRT_LOGGER)

# https://github.com/onnx/onnx-tensorrt/issues/180
def allocate_buffers(engine):
    """
    Allocates all buffers required for the specified engine
    """
    inputs = []
    outputs = []
    bindings = []
    # Iterate over binding names in engine
    for binding in engine:
        # Get binding (tensor/buffer) size
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        # Get binding (tensor/buffer) data type (numpy-equivalent)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate page-locked memory (i.e., pinned memory) buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        # Allocate linear piece of device memory
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings
        bindings.append(int(device_mem))
        # Append to inputs/ouputs list
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    # Create a stream (to eventually copy inputs/outputs and run inference)
    stream = cuda.Stream()
    return inputs, outputs, bindings, stream

def infer(context, bindings, inputs, outputs, stream, batch_size=1):
    """
    Infer outputs on the IExecutionContext for the specified inputs
    """
    # Transfer input data to the GPU
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return the host outputs
    return [out.host for out in outputs]

# https://github.com/NVIDIA/object-detection-tensorrt-example/blob/master/SSD_Model/utils/common.py
# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# dummy data
input_src = np.random.random(size=(1,3,1080,1920)).astype(np.float32)
input_bgr = np.random.random(size=(1,3,1080,1920)).astype(np.float32)

# Read the serialized ICudaEngine
with open(trt_engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    # Deserialize ICudaEngine
    engine = runtime.deserialize_cuda_engine(f.read())
# Now just as with the onnx2trt samples...
# Create an IExecutionContext (context for executing inference)
with engine.create_execution_context() as context:
    # Allocate memory for inputs/outputs
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    # Set host input to the image
    inputs[0].host = input_src
    inputs[1].host = input_bgr
    # Inference
    start_time = time.time()
    trt_outputs = infer(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    end_time = time.time() -start_time
    print("Inference Time[ms]", end_time*1000)
    # Prediction
    print(trt_outputs)

