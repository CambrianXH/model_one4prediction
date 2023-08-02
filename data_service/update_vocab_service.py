'''
Author: aiwenjie aiwenjie20@outlook.com
Date: 2022-11-04 19:56:57
LastEditors: aiwenjie aiwenjie20@outlook.com
LastEditTime: 2022-11-05 12:02:35
FilePath: /wangyue/model_one/data_service/update_vocab_service.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import pandas as pd
import argparse
import numpy as np
import time
from multiprocessing import Pool
from tqdm import tqdm
from model_one.utils.cfg_utils import *
from model_one.utils.common_utils import round_str

def update_vocab(vocab_file, data_file, unused_num=200, num_obs=3, num_lanes=6,clusters=1000,
                processors = 1, data_precision = 4,special_data_precision = 3, expand_scale = 1000):

    # 默认标识符
    default_vocab_flag = ['[CLS]', '[UNK]', '[PAD]', '[SEP]',
                          '[MASK]']
    # , '[S2S_SOS]', '[S2S_SOS]', '[S2S_CLS]', '[S2S_SEP]'
    # 打开文件vocab文件
    vocab_list = []

    with open(vocab_file, mode='a+') as f:
        for line in f:
            vocab_list.append(line)

        # 写入默认标识符
        for content in default_vocab_flag:
            if content not in vocab_list:
                f.write(content+'\n')

        # 写入unused位置
        for idx in range(unused_num):
            unused_flag = f'[unused{idx}]'
            if unused_flag not in vocab_list:
                f.write(unused_flag+'\n')

         # 写入图片centroid
        for idx in range(clusters):
            if idx not in vocab_list:
                f.write(str(idx)+'\n')

    global_scope = []
    # 打开训练预料
    start_time = time.time()

    data_info_df = pd.read_csv(data_file)
    print(f'get data spend {round((time.time() - start_time),2)}s')

    max_values = data_info_df.max(axis=0)
    min_values = data_info_df.min(axis=0)

    # 自车信息
    vel_x_scope_ego = [round(min_values['vel_x'], data_precision), round(max_values['vel_x'],data_precision)]
    acc_x_scope_ego = [round(min_values['acc_x'], data_precision), round(max_values['acc_x'], data_precision)]
    steer_angle_scope_ego = [round(min_values['steer_angle'], data_precision), round(max_values['steer_angle'], data_precision)]
    head_yaw_ang_scope_ego = [round(min_values['head_yaw_ang'], data_precision), round(max_values['head_yaw_ang'], data_precision)]
    waypoint_x_scope_ego = [round(min_values['waypoint_x'], data_precision), round(max_values['waypoint_x'], data_precision)]
    waypoint_y_scope_ego = [round(min_values['waypoint_y'], data_precision), round(max_values['waypoint_y'], data_precision)]

    scope_ego = [vel_x_scope_ego,acc_x_scope_ego,steer_angle_scope_ego,head_yaw_ang_scope_ego,waypoint_x_scope_ego,waypoint_y_scope_ego]
    global_scope.extend(scope_ego)

    multi_data_process(data_info_df,num_obs,num_lanes,vocab_list,global_scope,data_precision,special_data_precision)


def multi_data_process(data_info_df,num_obs, num_lanes,vocab_list,global_scope,
                      data_precision = 3,special_data_precision = 3):
    scope_obs = []
    str_list = []
    def merge(intervals):
        # 合并区间
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # 如果列表为空，或者当前区间与上一区间不重合，直接添加
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # 否则的话，我们就可以与上一区间进行合并
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged

    for idx, item in tqdm(data_info_df.iterrows(),total = data_info_df.shape[0], desc='process vocabulary content'):
        keys_obs = ['geo_x','geo_y','geo_z','distance','vel_x','vel_y','accel_x','accel_y','theta_angle']
        def get_scope(info,keys):
            scope = []
            for key in keys:
                start, end = float('inf'), float('-inf')
                for dic in info:
                    start = min(start,round(dic[key],data_precision))
                    end = max(end,round(dic[key],data_precision))
                scope.append([start,end])
            return scope

        # 障碍物信息: 障碍物类型、障碍物自身尺寸、相对主车距离、障碍物相对速度、绝对速度，障碍物偏转角、障碍物加速度
        obs_info_list = eval(item['obs_info'])
        # 按照距离升序排序，选距离最近的障碍物
        if obs_info_list:
            obs_info_list.sort(key=lambda x: x['distance'], reverse=False)

        # 只取前self.num_obs个障碍物信息
        if len(obs_info_list) > num_obs:
            obs_info_list = obs_info_list[:num_obs]

        if obs_info_list:
            scope_obs.extend(get_scope(obs_info_list,keys_obs))
            for dic in obs_info_list:
                str_list.append(dic['obs_type'])

        # 车道线信息: 车道线类型、车道线颜色、车道线相对主车的位置
        lane_info_list = eval(item['lane_info'])
        # 只取前self.num_lanes条车道线信息
        if len(lane_info_list) > num_lanes:
            lane_info_list = lane_info_list[:num_lanes]
            if lane_info_list:
                for i in range(len(lane_info_list)):
                    line_color = lane_info_list[i]['line_color']
                    line_type = lane_info_list[i]['line_type']
                    line_pos = lane_info_list[i]['line_pos']
                    str_list.extend([line_color, line_type, line_pos])

    global_scope.extend(scope_obs)
    global_scope = merge(global_scope)

    for start,end in global_scope:
        write_in = np.arange(start, end + 0.0001, 0.0001)
        print(start,end)
    write_in = [round_str(num,data_precision) for num in write_in]
    write_in.extend(str_list)

    with open(vocab_file, mode='a+') as f:
        # 写入词汇表
        for content in tqdm(write_in,total = len(write_in), desc='write vocabulary content'):
            if content not in vocab_list:
                vocab_list.append(content)
                f.write(str(content) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg_path",
        type=str,
        default="/data/wangyue/model_one/config_service/m1_config_v1.yaml",
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    args = parser.parse_args()
    cfg_path = args.cfg_path
    cfg = load_yaml2cfg(cfg_path)

    vocab_file = '/data/wangyue/model_one/config_service/bert-traj-vocab_v3_3.txt'
    data_file = cfg.DATA_SOURCE.TXT_PATH_FOR_VOCAB

    update_vocab(vocab_file, data_file,num_obs=cfg.DATA.NUM_OBS, num_lanes=cfg.DATA.NUM_LANES)
