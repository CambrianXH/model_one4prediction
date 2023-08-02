'''
Author: wangyue
Date: 2022-11-08 18:06:08
LastEditTime: 2022-11-09 13:49:59
Description:
# 通用model server模块，当前可以获取transformer base和plus版本
'''

import sys,os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from model_base.trm_model import TransForPreTrainingLossMask
from model_base.trm_model_plus import TransForPreTrainingLossMaskPlus

def get_trm_model(cfg):
    model = TransForPreTrainingLossMask(in_features_size = cfg.MODEL.PRETRAIN.IN_FEATURES_SIZE,
            out_features_size = cfg.MODEL.PRETRAIN.OUT_FEATURES_SIZE,d_model = cfg.MODEL.PRETRAIN.D_MODEL, 
            nhead = cfg.MODEL.PRETRAIN.N_HEADS, num_layers = cfg.MODEL.PRETRAIN.N_LAYERS, 
            dropout = cfg.MODEL.PRETRAIN.HIDDEN_DROPOUT_PROB,src_seq_len = cfg.DATA.FRAME.SRC_FRAME_LEN)
    return model

def get_trm_model_plus(cfg):
    model = TransForPreTrainingLossMaskPlus(in_features_size = cfg.MODEL.PRETRAIN.IN_FEATURES_SIZE,
            out_features_size = cfg.MODEL.PRETRAIN.OUT_FEATURES_SIZE,d_model = cfg.MODEL.PRETRAIN.D_MODEL, 
            nhead = cfg.MODEL.PRETRAIN.N_HEADS, num_layers = cfg.MODEL.PRETRAIN.N_LAYERS, 
            dropout = cfg.MODEL.PRETRAIN.HIDDEN_DROPOUT_PROB,num_embeddings = cfg.MODEL.PRETRAIN.EMB_SIZE,
            tgt_seq_len = cfg.DATA.FRAME.TGT_FRAME_LEN, is_img=cfg.DATA.IS_IMG, in_img_size=cfg.DATA.FRAME.IMG_FEATURES_LEN)
    return model
