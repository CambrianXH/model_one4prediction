"""
    ○ 功能：读取oss路径文件，加载模型，推理数据，写入一个文件【x、y、v、headng】
    ○ 参数：input_txt_path,input_img_path,output_data_path

"""

import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
sys.path.append("..")
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch
import copy
from utils.cfg_utils import *
from model_base.multi_model_encoder import *
from model_base.model_utils import *
from model_base.traj_evaluator import TrajEvaluator
from model_base.model_server import get_trm_model,get_trm_model_plus
from data_service.seq2seq_dataloader import *
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
from datetime import datetime


def main():
    cfg = load_yaml2cfg("/data/wangyue/model_one/config_service/m1_config_base.yaml")
    model = get_model(cfg=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取数据
    # data_txt_path = cfg.DATA_SOURCE.TXT_PATH
    data_txt_path = '/data/wangyue/model_one/tmp/turn_head.csv'
    txt_data_df = pd.read_csv(data_txt_path)
    loader_dict,nums_dict = get_data(cfg=cfg, df=txt_data_df)
    # 推理
    evaluator = Infer(model, loader_dict, nums_dict,cfg, device, save_path=cfg.EVAL.SAVE_PATH)
    evaluator.infer()      #basic


def get_model(cfg):
    model_state_dict = torch.load(cfg.EVAL.MODEL_DIR)
    # model = get_trm_model_plus(cfg)
    model = get_trm_model(cfg)
    model.load_state_dict({k.replace('module.', ''): v for k, v in model_state_dict.items()})
    return model

def get_data(cfg,df):
    data_info_df = df
    data_info_df['ts'] = data_info_df['ts'].astype(int)
    total_num = data_info_df.shape[0]
    # train_num = math.ceil(total_num * cfg.TRAIN.TRAIN_RATE)
    train_num = 0
    num_dict = {'train': train_num, 'valid': total_num - train_num}

    val_dataset = Seq2SeqEncoderDataset(
        data_df = data_info_df.iloc[train_num:],
        src_frame_len = cfg.DATA.FRAME.SRC_FRAME_LEN,
        tgt_frame_len = cfg.DATA.FRAME.TGT_FRAME_LEN,
        frame_step = cfg.DATA.FRAME.FRAME_STEP,
        is_only_waypoint = cfg.DATA.IS_ONLY_WAYPOINT,
        is_img = cfg.DATA.IS_IMG,
        img_feat_path = cfg.DATA_SOURCE.IMG_FEATURE_PATH,
        mode = 'test'
    )

    loader_val = DataLoader(
        dataset = val_dataset,
        batch_size = cfg.EVAL.BATCH_SIZE,
        # batch_size = 1,
        num_workers = cfg.EVAL.NUM_WORKS,
        drop_last = True
    )
    
    loader_dict = {'valid': loader_val}
    return loader_dict, num_dict

class Infer:
    def __init__(self, model, loader_dict, nums_dict, cfg, device, save_path):
        self.cfg = cfg
        self.model = copy.deepcopy(model)
        self.loader_dict = loader_dict
        self.nums_dict = nums_dict
        self.device = device
        self.save_path = save_path  # 输出保存路径
        self.info_size = cfg.MODEL.PRETRAIN.OUT_FEATURES_SIZE + 2
    
    def infer(self):
        '''
        1、推理：读batch推理数据，预测x、y、v、a等值
        2、保存数据：x/y/v/a等值
        '''
        val_dataloader = self.loader_dict['valid']
        self.model.to(self.device)
        self.model.eval()

        total_infos = np.array([])
        with torch.no_grad():
            for iter_idx, batch_data in enumerate(tqdm(val_dataloader,desc="eval batch")):
                src_seq,_,basic_info_list = batch_data[0].permute([1,0,2]).contiguous().to(self.device), \
                                                batch_data[1].to(self.device), batch_data[2][-self.cfg.DATA.FRAME.TGT_FRAME_LEN:]
                tgt_pred = self.model(src=src_seq)
                tgt_pred = tgt_pred[-self.cfg.DATA.FRAME.TGT_FRAME_LEN:].permute([1,0,2]).contiguous()[:,:,:2]
                
                for i in range(len(basic_info_list)):
                    basic_info_list[i][1] = basic_info_list[i][1].numpy()

                basic_info_list = np.array(basic_info_list).transpose(2,0,1)
                batch_size, *_ = tgt_pred.shape

                one_batch_infos = tgt_pred.cpu().numpy()
                one_batch_infos = np.concatenate([basic_info_list, one_batch_infos],axis=2)
                if total_infos.size == 0 :
                    total_infos = one_batch_infos
                else:
                    total_infos = np.append(total_infos, values=one_batch_infos,axis=0)
            total_infos = total_infos.reshape(batch_size,-1,self.info_size).reshape(-1,self.info_size) 
        if self.info_size == 6:
            df_traj = pd.DataFrame(total_infos,columns=['car','ts','x','y','v','a'])
        elif self.info_size == 4:
            df_traj = pd.DataFrame(total_infos,columns=['car','ts','x','y',])
        df_traj.to_csv(self.save_path,index=False,na_rep=0)
        # 将数据转成json
        self.to_json_and_save_data(df_traj)
        
        
    def to_json_and_save_data(self,df_traj):
        planning_output_list = {
            'trajectory_point':[]
        }
        step = 50
        for idx in range(0,df_traj.shape[0],step):
            for _, item in df_traj[idx:idx+step].iterrows():
                car = item['car']
                ts = item['ts']
                x = item['x']
                y = item['y']
                v = item.get('v',0.0)
                a = item.get('a',0.0)
                single_frame = {
                    'relative_time':0.1,
                    'a':a,
                    'v':v,
                    'path_point':{
                        'x':x,
                        'y':y,
                        'theta':0.0,
                        'kappa':0.0,
                    },
                }
                planning_output_list['trajectory_point'].append(single_frame)
            planning_output_json = json.dumps(planning_output_list)
            print(planning_output_json)
            planning_output_list['trajectory_point'] = []
            # save planning data to real path
            save_dir = os.path.join(self.cfg.DIR.INFER_DIR,self.cfg.EXP.MODEL_NAME,(datetime.now()).strftime("%Y-%m-%d-%H-%M-%S"))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir,exist_ok=True)
            with open(f'{save_dir}/{car}_{ts}_idx','w') as f:
                f.write(planning_output_json)
        




if __name__ == "__main__":
    main()




