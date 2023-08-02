'''
Author: wangyue
Date: 2022-08-24 11:20:32
LastEditors: aiwenjie aiwenjie20@outlook.com
LastEditTime: 2022-11-08 09:58:51
FilePath: /model_one/data_service/dataloader.py
Description: 	

Copyright (c) 2022 by Haomo, All Rights Reserved.
'''
import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from data_service.meta_data import *
from utils.common_utils import round_str
from utils.cfg_utils import *
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import time
import math
import numpy as np
import pandas as pd
import json

def get_data(cfg):
    start_time = time.time()
    data_txt_path = cfg.DATA_SOURCE.TXT_PATH
    data_info_df = pd.read_csv(data_txt_path)
    data_info_df['ts'] = data_info_df['ts'].astype(int)
    total_num = data_info_df.shape[0]
    train_num = math.ceil(total_num * cfg.TRAIN.TRAIN_RATE)
    val_num = total_num - train_num
    num_dict = {'train': train_num, 'valid': val_num}

    train_dataset = Seq2SeqEncoderDataset(
        data_df=data_info_df.iloc[:train_num],
        src_frame_len=cfg.DATA.FRAME.SRC_FRAME_LEN,
        tgt_frame_len=cfg.DATA.FRAME.TGT_FRAME_LEN,
        frame_step=cfg.DATA.FRAME.FRAME_STEP,
        is_only_waypoint=cfg.DATA.IS_ONLY_WAYPOINT,
        is_img=cfg.DATA.IS_IMG,
        img_feat_path=cfg.DATA_SOURCE.IMG_FEATURE_PATH
    )
    val_dataset = Seq2SeqEncoderDataset(
        data_df=data_info_df.iloc[train_num:],
        src_frame_len=cfg.DATA.FRAME.SRC_FRAME_LEN,
        tgt_frame_len=cfg.DATA.FRAME.TGT_FRAME_LEN,
        frame_step=cfg.DATA.FRAME.FRAME_STEP,
        is_only_waypoint=cfg.DATA.IS_ONLY_WAYPOINT,
        is_img=cfg.DATA.IS_IMG,
        img_feat_path=cfg.DATA_SOURCE.IMG_FEATURE_PATH
    )
    # DataLoader for Distributed DataParallel
    loader_train = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.MODEL.PRETRAIN.NUM_WORKS,
        sampler=DistributedSampler(train_dataset),
        drop_last=True
    )

    loader_val = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.EVAL.BATCH_SIZE,
        num_workers=cfg.MODEL.PRETRAIN.NUM_WORKS,
        sampler=DistributedSampler(val_dataset),
        drop_last=True
    )

    if dist.get_rank() == 0:
        print('Spliting dataset and getting dataloader spend {:.4f}s.' .format(
            time.time() - start_time))

    loader_dict = {'train': loader_train, 'valid': loader_val}
    return loader_dict, num_dict


class Seq2SeqEncoderDataset(Dataset):
    '''Dataset Encoder Class, return text token.'''

    def __init__(self, data_df, src_frame_len, tgt_frame_len, frame_step,
                 num_obs=3, num_lanes=4, data_precision=4, is_only_waypoint=True, 
                 is_img=False, img_feat_path=None,is_nav=True,is_goal=True,mode='train'):
        self.img_feat_path = img_feat_path
        self.mode = mode
        self.data_df = data_df
        self.src_frame_len = src_frame_len
        self.tgt_frame_len = tgt_frame_len if mode == 'train' else 0
        self.frame_step = frame_step
        self.num_obs = num_obs
        self.num_lanes = num_lanes
        self.data_precision = data_precision
        self.len = self.data_df.shape[0]
        self.is_nav = is_nav
        self.is_goal = is_goal
        self.obs_col_num = 10  # 障碍物字段数
        self.lane_col_num = 3  # 车道线字段数
        self.ego_col_num = 8  # 自车信息字段数
        self.goal_col_num = 2 if self.is_goal else 0 # 自车目标字段数
        self.seq_default_val = -1000.0
        self.is_only_waypoint = is_only_waypoint  # 未来仅预测未来轨迹点x/y
        self.is_img = is_img


    def __len__(self):
        return self.len - (self.src_frame_len + self.tgt_frame_len) + self.frame_step

    def __getitem__(self, idx):
        seq_list = self.get_seq_list(idx)
        return seq_list

    def get_seq_list(self, idx):
        seq_data_df = self.data_df[idx:idx + self.src_frame_len + self.tgt_frame_len]
        # 坐标点累加 start
        waypoint_x = np.asarray(seq_data_df['waypoint_x'].values)
        waypoint_y = np.asarray(seq_data_df['waypoint_y'].values)

        waypoint_x = np.cumsum(waypoint_x,axis=0)
        waypoint_y = np.cumsum(waypoint_y,axis=0) 

        tgt_first_frame_x,tgt_first_frame_y = waypoint_x[self.src_frame_len], waypoint_y[self.src_frame_len]

        waypoint_x -= tgt_first_frame_x
        waypoint_y -= tgt_first_frame_y

        seq_frame_data_df = seq_data_df.copy()
        seq_frame_data_df['waypoint_x'] = waypoint_x
        seq_frame_data_df['waypoint_y'] = waypoint_y
        # 坐标点累加 end
        src_frame_data_df = seq_frame_data_df[:self.src_frame_len:self.frame_step]
        tgt_frame_data_df = seq_frame_data_df[self.src_frame_len::self.frame_step]
        # print(src_frame_data_df)
        # print(tgt_frame_data_df)

        basic_seq_list = [] # basic information
        src_seq_list = [] # history information
        tgt_seq_list = [] # feature information
        goal_waypoint_x, goal_waypoint_y = 0, 0  # 未来轨迹最后一帧轨迹点x/y
        # 先遍历未来轨迹，找到最后一帧轨迹点x/y

        if self.tgt_frame_len > 0:
            idx = 1
            for _, item in tgt_frame_data_df.iterrows():
                waypoint_x = item['waypoint_x']
                waypoint_y = item['waypoint_y']
                if idx == (tgt_frame_data_df.shape[0]):
                    goal_waypoint_x = waypoint_x 
                    goal_waypoint_y = waypoint_y
                    break
                idx += 1

        # 遍历历史数据
        if self.src_frame_len > 0:
            global_idx = 1
            for _, item in src_frame_data_df.iterrows():
                # obs information
                # 障碍物信息: 障碍物类型、障碍物自身尺寸、相对主车距离、障碍物相对速度、绝对速度，障碍物偏转角、障碍物加速度
                obs_info_list = eval(item['obs_info'])
                # 按照距离升序排序，选距离最近的障碍物
                if not obs_info_list:
                    obs_info_list.sort(
                        key=lambda x: x['distance'], reverse=False)

                # 只取前self.num_obs个障碍物信息
                if len(obs_info_list) > self.num_obs:
                    obs_info_list = obs_info_list[:self.num_obs]

                src_seq_tensor = torch.zeros(
                    self.obs_col_num*self.num_obs + self.lane_col_num*self.num_lanes + self.ego_col_num + self.goal_col_num)
                # hist_seq_tensor = torch.full([self.obs_col_num*self.num_obs + self.lane_col_num*self.num_lanes + self.ego_col_num],self.seq_default_val)
                if obs_info_list:
                    for idx in range(len(obs_info_list)):
                        obs_type = obs_info_list[idx]['obs_type']
                        obs_type_id = obs_type_dict[obs_type]
                        obs_geo_x = obs_info_list[idx]['geo_x']
                        obs_geo_y = obs_info_list[idx]['geo_y']
                        obs_geo_z = obs_info_list[idx]['geo_z']
                        obs_distance = obs_info_list[idx]['distance']
                        obs_vel_x = obs_info_list[idx]['vel_x']
                        obs_vel_y = obs_info_list[idx]['vel_y']
                        obs_accel_x = obs_info_list[idx]['accel_x']  
                        obs_accel_y = obs_info_list[idx]['accel_y'] 
                        obs_theta_angle = obs_info_list[idx]['theta_angle']
 
                        obs_info_tensor = torch.tensor([obs_type_id,obs_geo_x,obs_geo_y,obs_geo_z,
                                                        obs_distance,obs_vel_x,obs_vel_y,obs_accel_x,
                                                        obs_accel_y,obs_theta_angle])
                        src_seq_tensor[idx*self.obs_col_num:(
                            idx+1)*self.obs_col_num] = obs_info_tensor

                # 车道线信息: 车道线类型、车道线颜色、车道线相对主车的位置
                lane_info_list = eval(item['lane_info'])
                # 只取前self.num_lanes条车道线信息
                if len(lane_info_list) > self.num_lanes:
                    lane_info_list = lane_info_list[:self.num_lanes]
                    if lane_info_list:
                        for i in range(len(lane_info_list)):
                            line_color = lane_info_list[i]['line_color']
                            line_color_id = line_color_dict[line_color]

                            line_type = lane_info_list[i]['line_type']
                            line_type_id = line_type_dict[line_type]

                            line_pos = lane_info_list[i]['line_pos']
                            line_pos_id = line_pos_dict[line_pos]

                            src_seq_tensor[self.obs_col_num*self.num_obs + self.lane_col_num*i: self.obs_col_num *
                                           self.num_obs + self.lane_col_num*(i+1)] = torch.tensor([line_color_id, line_type_id, line_pos_id])
                # ego information
                ego_info = json.loads(item['ego_info'])
                vel_x = ego_info['vel_x']
                vel_y = ego_info['vel_y']
                acc_x = ego_info['acc_x']
                acc_y = ego_info['acc_y']
                steer_angle = ego_info['steer_angle']
                head_yaw_ang = ego_info['euler_yaw']
                
                waypoint_x = item['waypoint_x']
                waypoint_y = item['waypoint_y']
                ego_motion = item['ego_motion'] # 中文
                ego_motion_idx = ego_motions[ego_motion]
                ego_info_tensor = torch.tensor(
                    [vel_x,vel_y, acc_x,acc_y, steer_angle, head_yaw_ang, waypoint_x, waypoint_y])
                src_seq_tensor[self.obs_col_num*self.num_obs + self.lane_col_num*self.num_lanes:self.obs_col_num *
                               self.num_obs + self.lane_col_num*self.num_lanes + self.ego_col_num] = ego_info_tensor
                if self.is_goal:
                # goal waypoint set to src last frame information
                    if global_idx == (src_frame_data_df.shape[0]):
                        goal_waypoint_tensor = torch.tensor(
                            [goal_waypoint_x, goal_waypoint_y])
                        src_seq_tensor[self.obs_col_num*self.num_obs + self.lane_col_num *
                                    self.num_lanes + self.ego_col_num:] = goal_waypoint_tensor
                    global_idx += 1

                if self.is_nav:
                    # nav goal information
                    goal_waypoint_tensor = torch.zeros(len(ego_motions))
                    nav_goal = ego_motion_idx
                    goal_waypoint_tensor[nav_goal-1] = 1 # 导航目标值从1开始
                    src_seq_tensor = torch.cat([src_seq_tensor, goal_waypoint_tensor])

                src_seq_list.append(src_seq_tensor)

                # basic information
                car_id = (item['car_id'])
                ts = (item['ts'])
                basic_seq_list.append([car_id, ts])


        # 遍历未来数据
        if self.tgt_frame_len > 0:
            for _, item in tgt_frame_data_df.iterrows():
                ego_info = json.loads(item['ego_info'])
                vel_x = ego_info['vel_x']
                vel_y = ego_info['vel_y']
                vel = math.sqrt(vel_x * vel_x + vel_y * vel_y) 
                acc_x = ego_info['acc_x']
                acc_y = ego_info['acc_y']
                acc = math.sqrt(acc_x * acc_x + acc_y * acc_y)
                # steer_angle = ego_info['steer_angle']# 当前不用预测方向盘转角
                head_yaw_ang = ego_info['euler_yaw']

                waypoint_x = item['waypoint_x']
                waypoint_y = item['waypoint_y']
                pred_tensor = None
                if self.is_only_waypoint:
                    pred_tensor = torch.tensor([waypoint_x, waypoint_y])
                else:
                    pred_tensor = torch.tensor(
                        [waypoint_x, waypoint_y, vel, acc, head_yaw_ang])
                tgt_seq_list.append(pred_tensor)
                # ego basic information
                car_id = (item['car_id'])
                ts = (item['ts'])
                basic_seq_list.append([car_id, ts])

        src_seq = torch.stack(src_seq_list, dim=0)
        tgt_seq = torch.stack(tgt_seq_list, dim=0) if len(tgt_seq_list) > 0 else torch.empty(1)
        if self.is_img:
            # 获取img
            img_files = [
                i[0] + '_' + str(i[1]) + '.npy' for i in basic_seq_list[:self.src_frame_len]]
            feat_path = [os.path.join(self.img_feat_path, i)
                         for i in img_files]
            img_features = [torch.from_numpy(
                np.load(path)) for path in feat_path]
            return src_seq, tgt_seq, basic_seq_list, torch.stack(img_features, dim=0).squeeze()
        
        else:
            return src_seq, tgt_seq, basic_seq_list

if __name__ == '__main__':
    cfg = load_yaml2cfg(
        "/data/wangyue/model_one/config_service/m1_config_base.yaml")
    loader_dict, nums_dict = get_data(cfg=cfg)
    val_loader = loader_dict['valid']
    val_loader[900]
    for idx, data in enumerate(val_loader):
        print(data)

