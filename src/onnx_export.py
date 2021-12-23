from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

def export(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  src_path = '../models/ctdet_coco_resdcn18.pth'

  dst_path = src_path.split("pth")[0] + "onnx"
  dummy_input = torch.randn(1, 3, 512, 512, requires_grad=True)

  model = detector.model
  # model.eval()
  model.to("cpu")
  # #import pdb
  # #pdb.set_trace()
  torch.onnx.export(model, dummy_input, "/home/jmye/test.onnx")


  # if opt.demo == 'webcam' or \
  #   opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
  #   cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
  #   detector.pause = False
  #   while True:
  #       _, img = cam.read()
  #       cv2.imshow('input', img)
  #       ret = detector.run(img)
  #       time_str = ''
  #       for stat in time_stats:
  #         time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
  #       print(time_str)
  #       if cv2.waitKey(1) == 27:
  #           return  # esc to quit
  # else:
  #   if os.path.isdir(opt.demo):
  #     image_names = []
  #     ls = os.listdir(opt.demo)
  #     for file_name in sorted(ls):
  #         ext = file_name[file_name.rfind('.') + 1:].lower()
  #         if ext in image_ext:
  #             image_names.append(os.path.join(opt.demo, file_name))
  #   else:
  #     image_names = [opt.demo]
    
  #   for (image_name) in image_names:
  #     ret = detector.run(image_name)
  #     time_str = ''
  #     for stat in time_stats:
  #       time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
  #     print(time_str)
