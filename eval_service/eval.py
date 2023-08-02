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
from eval_service.visz_traj import Visualization


def main():
    cfg = load_yaml2cfg("/data/wangyue/model_one/config_service/m1_config_base.yaml")
    model = get_model(cfg=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取数据
    data_txt_path = cfg.EVAL.DATA_PATH
    txt_data_df = pd.read_csv(data_txt_path)
    loader_dict,nums_dict = get_data(cfg=cfg, df=txt_data_df)
    data_basename = os.path.splitext(os.path.basename(data_txt_path))[0]
    # indicator file、output file、trajectory imgs file
    eval_output_dir = os.path.join(cfg.DIR.EVAL_DIR,cfg.EXP.MODEL_NAME,data_basename) 
    cfg.eval_output_dir = eval_output_dir
    
    # 推理
    evaluator = Eval(model, loader_dict, nums_dict,cfg, device, is_viz=cfg.EVAL.IS_VISZ)
    evaluator.eval()      #basic版本
    # 绘制gif
    if cfg.EVAL.IS_VISZ:
        visz_block = Visualization(cfg)
        output_traj_path = os.path.join(cfg.eval_output_dir,'output_traj.csv')
        visz_block.eval_plot(cfg, txt_data_df, output_traj_path)

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
        img_feat_path = cfg.DATA_SOURCE.IMG_FEATURE_PATH
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

class Eval:
    def __init__(self, model, loader_dict, nums_dict, cfg, device , is_viz=True):
        self.cfg = cfg
        self.model = copy.deepcopy(model)
        self.loader_dict = loader_dict
        self.nums_dict = nums_dict
        self.evalutor = TrajEvaluator()
        self.device = device
        self.is_viz = is_viz
        self.eval_output_dir = cfg.eval_output_dir
        if not os.path.exists(self.eval_output_dir):
            os.makedirs(self.eval_output_dir,exist_ok=True)
        self.indicator_path = os.path.join(self.eval_output_dir,'indicator_summary_info.txt')
        self.output_traj_path = os.path.join(self.eval_output_dir,'output_traj.csv')

    def eval(self):
        # 评估basic版本
        val_dataloader = self.loader_dict['valid']
        self.model.to(self.device)
        self.model.eval()

        total_infos = np.array([])
        with torch.no_grad():
            ade,fde,rmse = 0.0, 0.0, 0.0
            count = 0
            metric_infos = []
            total_ade_list = []
            for iter_idx, batch_data in enumerate(tqdm(val_dataloader,desc="eval batch")):
                src_seq, tgt_gt, basic_info_list = batch_data[0].permute([1,0,2]).contiguous().to(self.device), \
                                                batch_data[1].to(self.device), batch_data[2][-self.cfg.DATA.FRAME.TGT_FRAME_LEN:]
                tgt_pred = self.model(src=src_seq)
                tgt_pred = tgt_pred[-self.cfg.DATA.FRAME.TGT_FRAME_LEN:].permute([1,0,2]).contiguous()
            
                pred_waypoints, gt_waypoints = tgt_pred[:,:,:2], tgt_gt[:,:,:2]

                for i in range(len(basic_info_list)):
                    basic_info_list[i][1] = basic_info_list[i][1].numpy()
                basic_info_list = np.array(basic_info_list).transpose(2,0,1)
                
                batch_size, *_ = tgt_pred.shape
                
                metric = self.evalutor.compute_prediction_metrics(pred_waypoints.unsqueeze(1), gt_waypoints)   
                ade_list = self.evalutor.compute_batch_ade(pred_waypoints.unsqueeze(1), gt_waypoints)
                batch_rmse = self.evalutor.compute_rmse(tgt_pred,tgt_gt)
                total_ade_list.extend(ade_list)
                print(iter_idx,'\t',batch_rmse)
                metric_infos.append([metric["ADE"], metric["FDE"], batch_rmse])  #存储每个batch的ade

                ade += metric["ADE"]  #整批数据ade
                fde += metric["FDE"]
                rmse = (rmse * iter_idx + batch_rmse) / (iter_idx + 1)
                count += 1
                
                one_batch_infos = torch.cat([pred_waypoints, gt_waypoints],dim=2).cpu().numpy()
                one_batch_infos = np.concatenate([basic_info_list, one_batch_infos],axis=2)
                if total_infos.size == 0 :
                    total_infos = one_batch_infos
                else:
                    total_infos = np.append(total_infos, values=one_batch_infos,axis=0)

            summary_info = {'ADE':ade/count, 
                            'FDE':fde/count,
                            'RMSE':rmse}
            print("total ade:", summary_info['ADE'])
            print("total fde:", summary_info['FDE'])
            print("total rmse:", summary_info['RMSE'])
            total_infos = total_infos.reshape(batch_size,-1,6).reshape(-1,6) 
            metric_infos = np.array(metric_infos).reshape(-1,3)
        
        if self.is_viz:
            df_traj = pd.DataFrame(total_infos,columns=["car","ts","pred_x","pred_y","gt_x","gt_y"])
            df_traj['ADE'] = [item for i in total_ade_list for item in [i] * 50]
            df_traj.to_csv(self.output_traj_path,index=False,na_rep=0)

            # df_metric = pd.DataFrame(metric_infos,columns=["ADE","FDE","RMSE"])
            # df_metric.to_csv(self.ade_save_path,index=False)
            
            # eval summary 
            with open(self.indicator_path,'w') as f:
                f.write(json.dumps(summary_info))
            
    def eval_plus(self):
        # 评估plus版本
        val_dataloader = self.loader_dict['valid']
        self.model.to(self.device)
        self.model.eval()

        total_infos = np.array([])
        with torch.no_grad():
            ade,fde = 0.0,0.0
            count = 0
            metric_infos = []
            for iter_idx, batch_data in enumerate(val_dataloader):
                src_input, tgt_gt ,basic_info_list = batch_data[0].to(self.device), batch_data[1].to(self.device), \
                                                    batch_data[2][-self.cfg.DATA.FRAME.TGT_FRAME_LEN:]
                tgt_pred = self.get_forward_step_decode(self.cfg,self.model,src_input,[-1000,-1000])
               
                for i in range(len(basic_info_list)):
                    basic_info_list[i][1] = basic_info_list[i][1].numpy()
                basic_info_list = np.array(basic_info_list).transpose(2,0,1)

                batch_size, *_ = tgt_pred.shape
                metric = self.evalutor.compute_prediction_metrics(tgt_pred.unsqueeze(1), tgt_gt)

                metric_infos.append([metric["ADE"], metric["FDE"]])  #存储每个batch的ade
                ade += metric["ADE"]  #整批数据ade
                fde += metric["FDE"]
                count += 1
                
                one_batch_infos = torch.cat([tgt_pred, tgt_gt],dim=2).cpu().numpy()
                one_batch_infos = np.concatenate([basic_info_list, one_batch_infos],axis=2)
                if total_infos.size == 0 :
                    total_infos = one_batch_infos
                else:
                    total_infos = np.append(total_infos, values=one_batch_infos,axis=0)

            total_infos = total_infos.reshape(batch_size,-1,6).reshape(-1,6) 
            metric_infos = np.array(metric_infos).reshape(-1,2)

        if self.is_viz:
            df_traj = pd.DataFrame(total_infos,columns=["car","ts","pred_x","pred_y","gt_x","gt_y"])
            df_metric = pd.DataFrame(metric_infos,columns=["ADE","FDE"])
            df_traj.to_csv(self.save_path,index=False,na_rep=0)
            df_metric.to_csv(self.ade_save_path,index=False)
            print("write done")

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
    main()



