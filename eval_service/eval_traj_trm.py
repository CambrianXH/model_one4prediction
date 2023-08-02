'''
Author: wangyue
Date: 2022-11-08 18:06:08
LastEditTime: 2022-11-09 13:49:59
Description:
# 1、seq2seq_dataloader 改造decoder模块，跟encoder逻辑类似
# 2、加载预训练模型，模型文件：/data3/data_haomo/m1/pretrain/mmrt_v1.4/model.best_loss_1_epoch_1_eval.bin
# 3、轨迹预测[参考群里发的transformer.zip train.py文件]，并计算ADE值，复用traj_evaluator模块
'''


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
from model_one.utils.cfg_utils import *
from model_one.model_base.multi_model_encoder import *
from model_one.model_base.model_utils import *
from model_one.model_base.traj_evaluator import TrajEvaluator
from model_one.model_base.model_server import get_trm_model,get_trm_model_plus
from model_one.data_service.seq2seq_dataloader import *
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
from eval_service.visz_traj import Visualization


def get_model(cfg):
    model_state_dict = torch.load(cfg.EVAL.MODEL_DIR)
    # model = get_trm_model_plus(cfg)
    model = get_trm_model(cfg)
    model.load_state_dict({k.replace('module.', ''): v for k, v in model_state_dict.items()})
    return model

def get_data(cfg):
    # data_txt_path = cfg.DATA_SOURCE.TXT_PATH
    data_txt_path = '/data/wangyue/model_one/data_service/val_set.csv'
    n = 358
    data_info_df = pd.read_csv(data_txt_path)[n:n+130]
    data_info_df['ts'] = data_info_df['ts'].astype(int)
    total_num = data_info_df.shape[0]
    # train_num = math.ceil(total_num * cfg.TRAIN.TRAIN_RATE)
    train_num = 0
    val_num = total_num - train_num
    num_dict = {'train': train_num, 'valid': val_num}

    val_dataset = Seq2SeqEncoderDataset(
        data_df = data_info_df.iloc[train_num:],
        src_frame_len = cfg.DATA.FRAME.SRC_FRAME_LEN,
        tgt_frame_len = cfg.DATA.FRAME.TGT_FRAME_LEN,
        frame_step = cfg.DATA.FRAME.FRAME_STEP,
        is_only_waypoint = cfg.DATA.IS_ONLY_WAYPOINT,
        nav_goal_len = cfg.DATA.NAV_GOAL_LEN
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

def get_val_data(cfg,n,df):
    data_info_df = df[n:n+130]
    data_info_df['ts'] = data_info_df['ts'].astype(int)
    total_num = data_info_df.shape[0]

    train_num = 0
    val_num = total_num - train_num
    num_dict = {'train': train_num, 'valid': val_num}

    val_dataset = Seq2SeqEncoderDataset(
        data_df = data_info_df.iloc[train_num:],
        src_frame_len = cfg.DATA.FRAME.SRC_FRAME_LEN,
        tgt_frame_len = cfg.DATA.FRAME.TGT_FRAME_LEN,
        frame_step = cfg.DATA.FRAME.FRAME_STEP,
        is_only_waypoint = cfg.DATA.IS_ONLY_WAYPOINT,
        nav_goal_len = cfg.DATA.NAV_GOAL_LEN
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

def get_eval_id(path):
        id = pd.read_csv(path)['批次号'][:199].values.astype(int) * 130
        return id

class Eval:
    def __init__(self, model, loader_dict, nums_dict, cfg, device, traj_csv_name , is_viz=True):
        self.cfg = cfg
        self.model = copy.deepcopy(model)
        self.loader_dict = loader_dict
        self.nums_dict = nums_dict
        self.evalutor = TrajEvaluator()
        self.device = device
        self.is_viz = is_viz
        self.traj_csv_name = traj_csv_name


    def eval(self):
        val_dataloader = self.loader_dict['valid']
        self.model.to(self.device)
        self.model.eval()

        total_infos = np.array([])
        with torch.no_grad():
            ade,fde = 0.0,0.0
            count = 0
            metric_infos = []
            cfg.visual_dir = os.path.join(cfg.DIR.VISUAL, cfg.EXP.MODEL_NAME, 'visz/valid')
            visual_utils = VisualizationUtils(cfg)
            for iter_idx, batch_data in enumerate(val_dataloader):
                src_seq,tgt_gt,basic_info_list = batch_data[0].permute([1,0,2]).contiguous().to(self.device), \
                                                batch_data[1].to(self.device), batch_data[2][-self.cfg.DATA.FRAME.TGT_FRAME_LEN:]
                tgt_pred = self.model(src=src_seq)
                tgt_pred = tgt_pred[-self.cfg.DATA.FRAME.TGT_FRAME_LEN:].permute([1,0,2]).contiguous()[:,:,:2]
                # tgt_pred = tgt_pred[-47:].permute([1,0,2]).contiguous()[:,:,:2]

                for i in range(len(basic_info_list)):
                    basic_info_list[i][1] = basic_info_list[i][1].numpy()

                basic_info_list = np.array(basic_info_list).transpose(2,0,1)
                batch_size,num_frames, _ = tgt_pred.shape
                metric = self.evalutor.compute_prediction_metrics(tgt_pred.unsqueeze(1), tgt_gt)

                metric_infos.append([metric["ADE"], metric["FDE"]])  #存储每个batch的ade
                ade += metric["ADE"]  #整批数据ade
                fde += metric["FDE"]
                count += 1
                print(f"batch_ids:{iter_idx},metric:{metric}")
                
                valid_metrics_dict = {
                    'ade': metric["ADE"],
                    'fde': metric["FDE"],
                }
                visual_utils.record_metrics(iter_idx, None,valid_metrics_dict)
                
                one_batch_infos = torch.cat([tgt_pred, tgt_gt, tgt_pred - tgt_gt],dim=2).cpu().numpy()
                one_batch_infos = np.concatenate([basic_info_list, one_batch_infos],axis=2)
                if total_infos.size == 0 :
                    total_infos = one_batch_infos
                else:
                    total_infos = np.append(total_infos, values=one_batch_infos,axis=0)

            total_infos = total_infos.reshape(batch_size,-1,8).reshape(-1,8) 
            metric_infos = np.array(metric_infos).reshape(-1,2)
            print("Done","\n",f"info:  ADE:{ade/count},FDE:{fde/count}")
            visual_utils.manual_close_tb_writer() # 手动关闭tensorboard
        # out_path = os.path.join(cfg.DIR.VISUAL, cfg.EXP.MODEL_NAME, "visz", self.traj_csv_name)
        out_path = os.path.join('/data/wangyue/model_one/experiment/mmrt_v1.1/visz/',self.traj_csv_name)
        # ade_path = os.path.join(cfg.DIR.VISUAL, cfg.EXP.MODEL_NAME, "visz", "test.csv")
        ade_path = os.path.join('/data/wangyue/model_one/experiment/mmrt_v1.1/visz/',"test_ade.csv")

        if self.is_viz:
            df_traj = pd.DataFrame(total_infos,columns=["car","ts","pred_x","pred_y","gt_x","gt_y","diff_x","diff_y"])
            df_metric = pd.DataFrame(metric_infos,columns=["ADE","FDE"])
            df_traj.to_csv(out_path,index=False,na_rep=0)
            df_metric.to_csv(ade_path,index=False)
            print("write done")

    def eval_test(self):
        val_dataloader = self.loader_dict['valid']
        self.model.to(self.device)
        self.model.eval()

        total_infos = np.array([])
        with torch.no_grad():
            ade,fde = 0.0,0.0
            count = 0
            metric_infos = []
            cfg.visual_dir = os.path.join(cfg.DIR.VISUAL, cfg.EXP.MODEL_NAME, 'visz/valid')
            visual_utils = VisualizationUtils(cfg)
            for iter_idx, batch_data in enumerate(val_dataloader):
                src_seq,tgt_gt,basic_info_list = batch_data[0].permute([1,0,2]).contiguous().to(self.device), \
                                                batch_data[1].to(self.device), batch_data[2][-self.cfg.DATA.FRAME.TGT_FRAME_LEN:]
                tgt_pred = self.model(src=src_seq)
                tgt_pred = tgt_pred[-self.cfg.DATA.FRAME.TGT_FRAME_LEN:].permute([1,0,2]).contiguous()[:,:,:2]
                # tgt_pred = tgt_pred[-47:].permute([1,0,2]).contiguous()[:,:,:2]

                for i in range(len(basic_info_list)):
                    basic_info_list[i][1] = basic_info_list[i][1].numpy()

                basic_info_list = np.array(basic_info_list).transpose(2,0,1)
                batch_size,num_frames, _ = tgt_pred.shape
                metric = self.evalutor.compute_prediction_metrics(tgt_pred.unsqueeze(1), tgt_gt)

                metric_infos.append([metric["ADE"], metric["FDE"]])  #存储每个batch的ade
                ade += metric["ADE"]  #整批数据ade
                fde += metric["FDE"]
                count += 1            
                one_batch_infos = torch.cat([tgt_pred, tgt_gt, tgt_pred - tgt_gt],dim=2).cpu().numpy()
                one_batch_infos = np.concatenate([basic_info_list, one_batch_infos],axis=2)
                if total_infos.size == 0 :
                    total_infos = one_batch_infos
                else:
                    total_infos = np.append(total_infos, values=one_batch_infos,axis=0)

            total_infos = total_infos.reshape(batch_size,-1,8).reshape(-1,8) 
            metric_infos = np.array(metric_infos).reshape(-1,2)

                # out_path = os.path.join(cfg.DIR.VISUAL, cfg.EXP.MODEL_NAME, "visz", self.traj_csv_name)
        out_path = os.path.join('/data/wangyue/model_one/experiment/mmrt_v1.1/visz/',self.traj_csv_name)
        # ade_path = os.path.join(cfg.DIR.VISUAL, cfg.EXP.MODEL_NAME, "visz", "test.csv")
        ade_path = os.path.join('/data/wangyue/model_one/experiment/mmrt_v1.1/visz/',"test_ade.csv")

        if self.is_viz:
            df_traj = pd.DataFrame(total_infos,columns=["car","ts","pred_x","pred_y","gt_x","gt_y","diff_x","diff_y"])
            df_metric = pd.DataFrame(metric_infos,columns=["ADE","FDE"])
            df_traj.to_csv(out_path,index=False,na_rep=0)
            df_metric.to_csv(ade_path,index=False)
            print("write done")
        return ade/count

    def ddp_eval(self):
        val_dataloader = self.loader_dict['valid']
        self.model.to(self.device)
        self.model.eval()

        total_infos = np.array([])
        with torch.no_grad():
            ade,fde = 0.0,0.0
            count = 0
            metric_infos = []
            for iter_idx, batch_data in enumerate(val_dataloader):
                src_input,tgt_gt ,basic_info_list = batch_data[0].to(device), batch_data[1].to(device), batch_data[2][-self.cfg.DATA.FRAME.TGT_FRAME_LEN:]

                tgt_pred = self.get_forward_step_decode(self.cfg,self.model,src_input,[-1000,-1000])
               
                for i in range(len(basic_info_list)):
                    basic_info_list[i][1] = basic_info_list[i][1].numpy()

                basic_info_list = np.array(basic_info_list).transpose(2,0,1)
                batch_size,num_frames, _ = tgt_pred.shape
                metric = self.evalutor.compute_prediction_metrics(tgt_pred.unsqueeze(1), tgt_gt)

                metric_infos.append([metric["ADE"], metric["FDE"]])  #存储每个batch的ade
                ade += metric["ADE"]  #整批数据ade
                fde += metric["FDE"]
                count += 1
                print(f"batch_ids:{iter_idx},metric:{metric}")
                
                one_batch_infos = torch.cat([tgt_pred, tgt_gt, tgt_pred - tgt_gt],dim=2).cpu().numpy()
                one_batch_infos = np.concatenate([basic_info_list, one_batch_infos],axis=2)
                if total_infos.size == 0 :
                    total_infos = one_batch_infos
                else:
                    total_infos = np.append(total_infos, values=one_batch_infos,axis=0)

            total_infos = total_infos.reshape(batch_size,-1,8).reshape(-1,8) 
            metric_infos = np.array(metric_infos).reshape(-1,2)
            print("Done","\n",f"info:  ADE:{ade/count},FDE:{fde/count}")

        # out_path = os.path.join(cfg.DIR.VISUAL, cfg.EXP.MODEL_NAME, "visz", self.traj_csv_name)
        out_path = os.path.join('/data/wangyue/model_one/experiment/mmrt_v1.1/visz/',self.traj_csv_name)
        # ade_path = os.path.join(cfg.DIR.VISUAL, cfg.EXP.MODEL_NAME, "visz", "test.csv")
        ade_path = os.path.join('/data/wangyue/model_one/experiment/mmrt_v1.1/visz/',"test_ade.csv")

        if self.is_viz:
            df_traj = pd.DataFrame(total_infos,columns=["car","ts","pred_x","pred_y","gt_x","gt_y","diff_x","diff_y"])
            df_metric = pd.DataFrame(metric_infos,columns=["ADE","FDE"])
            df_traj.to_csv(out_path,index=False,na_rep=0)
            df_metric.to_csv(ade_path,index=False)
            print("write done")

    def ddp_eval_test(self):
        val_dataloader = self.loader_dict['valid']
        self.model.to(self.device)
        self.model.eval()

        total_infos = np.array([])
        with torch.no_grad():
            ade,fde = 0.0,0.0
            count = 0
            metric_infos = []
            for iter_idx, batch_data in enumerate(val_dataloader):
                src_input,tgt_gt ,basic_info_list = batch_data[0].to(device), batch_data[1].to(device), batch_data[2][-self.cfg.DATA.FRAME.TGT_FRAME_LEN:]

                tgt_pred = self.get_forward_step_decode(self.cfg,self.model,src_input,[-1000,-1000])
               
                for i in range(len(basic_info_list)):
                    basic_info_list[i][1] = basic_info_list[i][1].numpy()

                basic_info_list = np.array(basic_info_list).transpose(2,0,1)
                batch_size,num_frames, _ = tgt_pred.shape
                metric = self.evalutor.compute_prediction_metrics(tgt_pred.unsqueeze(1), tgt_gt)

                metric_infos.append([metric["ADE"], metric["FDE"]])  #存储每个batch的ade
                ade += metric["ADE"]  #整批数据ade
                fde += metric["FDE"]
                count += 1        
                one_batch_infos = torch.cat([tgt_pred, tgt_gt, tgt_pred - tgt_gt],dim=2).cpu().numpy()
                one_batch_infos = np.concatenate([basic_info_list, one_batch_infos],axis=2)
                if total_infos.size == 0 :
                    total_infos = one_batch_infos
                else:
                    total_infos = np.append(total_infos, values=one_batch_infos,axis=0)

            total_infos = total_infos.reshape(batch_size,-1,8).reshape(-1,8) 
            metric_infos = np.array(metric_infos).reshape(-1,2)

        # out_path = os.path.join(cfg.DIR.VISUAL, cfg.EXP.MODEL_NAME, "visz", self.traj_csv_name)
        out_path = os.path.join('/data/wangyue/model_one/experiment/mmrt_v1.1/visz/',self.traj_csv_name)
        # ade_path = os.path.join(cfg.DIR.VISUAL, cfg.EXP.MODEL_NAME, "visz", "test.csv")
        ade_path = os.path.join('/data/wangyue/model_one/experiment/mmrt_v1.1/visz/',"test_ade.csv")

        if self.is_viz:
            df_traj = pd.DataFrame(total_infos,columns=["car","ts","pred_x","pred_y","gt_x","gt_y","diff_x","diff_y"])
            df_metric = pd.DataFrame(metric_infos,columns=["ADE","FDE"])
            df_traj.to_csv(out_path,index=False,na_rep=0)
            df_metric.to_csv(ade_path,index=False)
        print("write done")
        return ade/count

    def get_forward_step_decode(self, cfg, model, enc_input, start_symbol):
        enc_outputs = model.encode_src(enc_input)
        next_symbol = torch.tensor(start_symbol).view(1,1,2).repeat(cfg.EVAL.BATCH_SIZE,1,1)
        dec_input = next_symbol.type_as(enc_input.data)
        for i in range(0,cfg.DATA.FRAME.TGT_FRAME_LEN):
            dec_outputs = model.decode_tgt(tgt=dec_input,memory=enc_outputs)
            next_symbol = dec_outputs[:,-1,:].unsqueeze(1)
            dec_input = torch.cat([dec_input,next_symbol],dim=1)
        output = model((enc_input,dec_input[:,:-1,:]))
        return output
                

if __name__ == "__main__":
    cfg = load_yaml2cfg("/data/wangyue/model_one/config_service/m1_config_base.yaml")
    # loader_dict,nums_dict = get_data(cfg=cfg)
    model = get_model(cfg=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # evaluator = Eval(model, loader_dict, nums_dict,cfg, device, traj_csv_name="test0103.csv")

    # evaluator.eval()   # 评估basic版本
    # evaluator.ddp_eval()  # 评估plus版本
    tmp = True

    if cfg.EVAL.IS_VISZ:
        visz_utils = Visualization(cfg,traj_csv_name="viz_data_v4.csv")

    if tmp:
        eval_path = '/data/wangyue/model_one/tmp/right.csv'
        eval_ids = get_eval_id(eval_path)
        ade = 0
        data_txt_path = '/data/wangyue/model_one/tmp/val_set.csv'
        txt_data_df = pd.read_csv(data_txt_path)
        for i in tqdm(eval_ids):
            loader_dict,nums_dict = get_val_data(cfg=cfg,n=i,df=txt_data_df)
            evaluator = Eval(model, loader_dict, nums_dict,cfg, device, traj_csv_name="test0103.csv",is_viz=cfg.EVAL.IS_VISZ)
            cur_ade = evaluator.eval_test()      #basic
            # cur_ade = evaluator.ddp_eval_test()  #plus
            print('id:',i,'\t',cur_ade)
            ade += cur_ade
            if cfg.EVAL.IS_VISZ:
                visz_utils.plot_src(n=i//130)
        ade /= len(eval_ids)
        print('ptrt_v1.6 200右变道样本ade:',ade)



