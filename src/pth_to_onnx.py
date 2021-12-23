
# # from torchreid.utils import FeatureExtractor
from onnx_export import *

# src_path = './log/truen_osnet_x05_market1501_cosine_fulltrans/model/model.pth.tar-180'
src_path = '../models/ctdet_coco_resdcn18.pth'

dst_path = src_path.split("pth.tar")[0] + "onnx"
dummy_input = torch.randn(1, 3, 512, 512, requires_grad=True)

opt = opts().init()
export(opt)


# # extractor = FeatureExtractor(
# #     model_name='osnet_x0_5',
# #     model_path=src_path,
# #     device='cpu'
# # )

# # model = extractor.model
# # torch.onnx.export(model, dummy_input, dst_path)

print("done")
