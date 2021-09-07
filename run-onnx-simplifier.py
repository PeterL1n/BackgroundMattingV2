import glob
import os
import subprocess

"""
Example:
    # Molding a fixed resolution with onnx-simpler
    # Process required to convert to TensorRT when model-refine-mode is FULL.

    pip3 install onnx-simplifier
    python3 -m onnxsim export_model.onnx export_model-simplifier.onnx

    python3 run-onnx-simplifier.py
"""

def gen_output_path(INPUT_PTH):
    # Generate output path

    dirname = os.path.dirname(INPUT_PTH)
    basename_without_ext = os.path.splitext(os.path.basename(INPUT_PTH))[0]
    ext = os.path.splitext(os.path.basename(INPUT_PTH))[1]
    print(dirname, basename_without_ext)
    MODEL_NAME  = basename_without_ext + "-simplifier" + ext
    print (MODEL_NAME)
    return MODEL_NAME


def run_rvm_cmd(INPUT_PTH):

    # Generate output path
    MODEL_NAME = gen_output_path(INPUT_PTH)
    print ("INPUT_PTH:", INPUT_PTH)
    print ("MODEL_NAME:", MODEL_NAME)

    # run RVM_cmd
    result = subprocess.call(["python3", "-m", "onnxsim", INPUT_PTH, MODEL_NAME])
    return result


if __name__ == '__main__':
    

    FILES_DIR = "./"

    # test_list = [*glob.glob(os.path.join(FILES_DIR, '**', '*.onnx'), recursive=True)]
    # print(test_list, len(test_list))

    # Get All File Path
    file_list = sorted([*glob.glob(os.path.join(FILES_DIR, '**', '*.onnx'), recursive=True)])
    print(file_list, len(file_list))

    #Loop
    for idx, video_file in enumerate(file_list):
        print(idx, "/", len(file_list))
        result = run_rvm_cmd(video_file)




