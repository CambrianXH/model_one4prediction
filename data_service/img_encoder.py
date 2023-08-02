'''
Author: aiwenjie aiwenjie20@outlook.com
Date: 2022-10-24 16:31:04
LastEditors: aiwenjie aiwenjie20@outlook.com
LastEditTime: 2022-11-07 15:49:31
FilePath: /model_one/model_base/img_encoder.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
@Project: model_one
@File: img_encoder.py
@Author: zhuliangqin
@Date: 2022/6/27 16:28
@LastEditTime: 2022/6/27 16:28
@Description: Encode a multi-dimensional image into one dimension

@License: Copyright (c) 2022 by Haomo, All Rights Reserved.
"""
import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from efficientnet_pytorch import EfficientNet
from torch.autograd import Variable
import PIL.Image as Image
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch import Tensor
import torchvision.models
import torch.nn as nn
import torch
import numpy as np
from utils.cfg_utils import *
import threading

def check_dirs(img_features_dir=None):
    if not os.path.exists(img_features_dir):
        os.makedirs(img_features_dir, exist_ok=True)


class ImgEncoder(nn.Module):
    '''use EfficientNet to encode current image information'''

    def __init__(self, output_dim: int, dropout: float, backbone='EfficientNet_b0', is_combined=False):
        '''
        @param output_dim: 输出维度
        @param is_combined: 六个视角图像是否拼接
        '''
        super().__init__()

        if backbone == 'EfficientNet_b0':
            self.model = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT).features[:7]
        elif backbone == 'EfficientNet_b4':
            self.model = torchvision.models.efficientnet_b4(
                weights=torchvision.models.EfficientNet_B4_Weights.DEFAULT)
        else:
            raise RuntimeError(
                'Please check config.MODEL.IMG.CUR_ENCODER, it must be EfficientNet_b0 or EfficientNet_b4.')
        self.reduction = nn.Conv2d(
            in_channels=192, out_channels=960, kernel_size=1, stride=1, padding=0)

    def forward(self, input: Tensor) -> Tensor:
        output = self.model(input)
        output = self.reduction(output)
        return output


def img_to_vec_v2(efficent_model_path, src_path):
    # 创建模型
    model = EfficientNet.from_name("efficientnet-b4")
    model.load_state_dict(torch.load(efficent_model_path,map_location=torch.device('cpu')))
    model.to('cpu')
    model.eval()

    # 将图片转为向量
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # efficient-b4, should be 380
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    features = []
    path_files = []
    for root, dirs, files in os.walk(src_path):
        for name in files:
            path_files.append(os.path.join(root, name))
    
    for file_name in (path_files):
        try:
            img = transform(Image.open(file_name).convert("RGB"))
            # print("open", time.time()-start_time)
        except Exception as err:
            print("[Error] open image", err)

        # 获取图片名称
        x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)

        try:
            efficient_feature = model.extract_features(x)
            efficient_feature = model._avg_pooling(
                efficient_feature).squeeze(3).squeeze(2)  # 1792
            numpy_feature = efficient_feature.cpu().detach()
            features.append(numpy_feature)
        except Exception as err:
            print("[Error] extract image", err)
        # np.save(os.path.join(save_path, os.path.splitext(os.path.basename(file_name))[0]), numpy_feature)
    return features

def get_feat(efficent_model_path,path_files,save_path):
     # 创建模型
    model = EfficientNet.from_name("efficientnet-b4")
    model.load_state_dict(torch.load(efficent_model_path))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.EXP.GPUS
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model.to('cpu')
    model.eval()
    
    # 将图片转为向量
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # efficient-b4, should be 380
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for file_name in path_files:
        try:
            img = transform(Image.open(file_name).convert("RGB"))
        except Exception as err:
            print("[Error] open image", err)
        # 获取图片名称
        x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False) # torch.Size([1, 3, 224, 224])
        try:
            efficient_feature = model.extract_features(x)
            efficient_feature = model._avg_pooling(
                efficient_feature).squeeze(3).squeeze(2)  # 1792
            numpy_feature = efficient_feature.cpu().detach().numpy()
        except Exception as err:
            print("[Error] extract image", err)
        np.save(os.path.join(save_path, os.path.splitext(
            os.path.basename(file_name))[0]), numpy_feature)



def multi_thread_img_to_vec(efficent_model_path,src_path, save_path, cfg):
    path_files = []
    # 获取所有图片的路径
    for root, dirs, files in os.walk(src_path):
        for name in files:
            path_files.append(os.path.join(root, name))
    total = len(path_files)
    idx = list(range(0,total,total//5))
    idx[-1] = total - 1

    threads = []
    for i in range(1,len(idx)):
        threads.append(
            threading.Thread(target=get_feat,
                            args=(efficent_model_path,tqdm(path_files[idx[i-1]:idx[i]],desc='thread {}'.format(i)),save_path))
        )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    cfg_path = '/data/wangyue/model_one/config_service/m1_config_plus.yaml'
    cfg = load_yaml2cfg(cfg_path)
    efficent_model_path = cfg.DATA_SOURCE.PRETRAINED_EFFINET_MODEL_PATH
    '/data/wangyue/model_one/pretrained_model/efficientnet-b4-6ed6700e.pth'
    src_path = '/data3/data_haomo/m1/bev/1212/pretrain_lane_change/'
    save_path = '/data3/data_haomo/m1/bev/1212/img_features'
    multi_thread_img_to_vec(efficent_model_path,src_path, save_path, cfg)
    # img_to_vec_v2(efficent_model_path,src_path)
