from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

import torch
import torch.onnx
import onnx

opt = opts().init()
print(" - arch : ", opt.arch)
print(" - load_model : ", opt.load_model)

dst_path = opt.load_model.replace(".pth", ".onnx")

# import pdb; pdb.set_trace()

Detector = detector_factory[opt.task]
detector = Detector(opt)

dummy_input = torch.randn(1, 3, 512, 512, requires_grad=True).to("cpu")
model = detector.model.to("cpu")
model.eval()

# torch.onnx
# torch.onnx.export(model, dummy_input, dst_path)

# onnx
torch.onnx.export(model, dummy_input, dst_path,
       verbose=False, opset_version=11,
       export_params=True, do_constant_folding=True)

print('Checking model...')
onnx_model = onnx.load(dst_path)
onnx.checker.check_model(onnx_model)

print("done")