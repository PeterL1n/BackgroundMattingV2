import onnx
import torch

onnx_model = onnx.load("make_onnx.onnx")
graph = onnx_model.graph
#print(graph)
#for tensor_info in [graph.node[401]]:
#    att= tensor_info.attribute
#    for at in att:
#        print('get',getattr(at, "i", None))
#        #setattr(at, "i",7)
#        print('get',getattr(at, "i", None))

#onnx.save(onnx_model, 'test_tr_opt.onnx')

src = torch.randn(1, 3, 1080, 1920)
src_bai = torch.mul(src, 10)
ref = src.gt(1).float()

print (src.shape)
print (src_bai.shape)
print (ref.shape)
