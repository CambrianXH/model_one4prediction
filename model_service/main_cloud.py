'''
Author: wangyue
Date: 2022-07-18 16:04:00
LastEditors: aiwenjie aiwenjie20@outlook.com
LastEditTime: 2022-11-08 11:20:25
FilePath: /model_one/model_service/main.py
Description: 	

Copyright (c) 2022 by Haomo, All Rights Reserved.
'''
import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from data_service.seq2seq_dataloader import get_data
from model_base.model_server import get_trm_model
from model_service.train import Trainer
from utils.cfg_utils import *

def get_model(cfg):
    start_time = time.time()
    model = get_trm_model(cfg)
    device = int(os.environ['LOCAL_RANK'])
    model = DDP(model.to(device), device_ids=[
                device], output_device=device, find_unused_parameters=True)
    if dist.get_rank() == 0:
        print('Model initialization cost {:.4f}s.'.format(time.time() - start_time))
    return model
    


def pretrain_traj_model_main(cfg):
    """
    预训练轨迹预测模型
    """
    if cfg.EXP.IS_MANUAL_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.EXP.GPUS
    # local_rank = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl') 

    # 获取数据的loader
    loader_dict,nums_dict = get_data(
        cfg=cfg,
    )

    model = get_model(cfg)

    trainer = Trainer(
        model=model,
        cfg=cfg,
        loader_dict=loader_dict,
        nums_dict=nums_dict,
    )

    trainer.ddp_train()


if __name__ == '__main__':
    cfg_file = os.path.join(project_path,"config_service/m1_config_cloud.yaml")
    cfg = load_yaml2cfg(cfg_file)
    pretrain_traj_model_main(cfg)


    
