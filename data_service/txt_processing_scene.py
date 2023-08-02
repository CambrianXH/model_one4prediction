'''
Author: aiwenjie aiwenjie20@outlook.com
Date: 2022-10-18 11:33:47
LastEditors: aiwenjie aiwenjie20@outlook.com
LastEditTime: 2022-11-07 16:25:51
FilePath: /model_one/data_service/text_processing.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: wangyue
Date: 022-06-23 19:13:45
LastEditors: aiwenjie aiwenjie20@outlook.com
LastEditTime: 2022-10-22 13:26:49
FilePath: /model_one/data_service/text_processing.py
Description: text information preprocessing.

Copyright (c) 2022 by Haomo, All Rights Reserved.
'''
import sys
sys.path.append('/data3/data_haomo/project/data-service')
from common.util.db.postgres_helper import get_postgres_helper
from fileinput import filename
from tkinter import Frame
import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import math
import sys
import csv

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)


class InfoProcessor:
    '''process columns info'''

    def __init__(self, csv_path,centroid_path):
        '''
        text_path: the path which saves chinese text.
        num_lanes: use the max items of lane info.
        num_obs: use the max items of obs info.
        '''
        self.csv_path = csv_path
        self.centroid_path = centroid_path
        with open(centroid_path, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)
    
    def convert_info(self, columns_info):
        ''''convert columns info to text info
        @columns_info: dataframe data's every column information, as a parameter for pandas.apply.
        '''
        fps_sin_dis_yaw = math.sin(columns_info['euler_yaw'])
        fps_cos_dis_yaw = math.cos(columns_info['euler_yaw'])
        dis_x = columns_info['vel_x']*0.1*fps_cos_dis_yaw
        dis_y = columns_info['vel_x']*0.1*fps_sin_dis_yaw
        
        ego_info = [{'steer_angle': columns_info['steer_angle'], 'vel_x':columns_info['vel_x'],
                     'accel_x':columns_info['accel_x'], 'euler_yaw':columns_info['euler_yaw'], 'dis_x':dis_x, 'dis_y':dis_y}]
        ego_info = json.dumps(ego_info)

        # 车道线信息: 车道线类型、车道线颜色、车道线相对主车的位置
        lane_info = columns_info['lane_info']

        # 障碍物信息: 障碍物类型、障碍物自身尺寸、相对主车距离、障碍物相对速度、绝对速度，障碍物偏转角、障碍物加速度
        obs_info = columns_info['obs_info']

        # 车辆ID信息
        file_name = columns_info['car_id'] + \
            '_' + str(columns_info['curr_ts'])

        nav_goal = columns_info['nav_goal']
        centroid = self.json_data.get(file_name,0)
        frame = pd.DataFrame({'car_id': columns_info['car_id'], 'ts': columns_info['curr_ts'], 'vel_x': columns_info['vel_x'], 'acc_x': columns_info['accel_x'], 'steer_angle': columns_info['steer_angle'],
                                 'head_yaw_ang': columns_info['euler_yaw']*0.1, 'obs_info': obs_info, 'lane_info': lane_info, 'centroid': centroid, 'file_name': file_name, 'waypoint_x': dis_x, 'waypoint_y': dis_y,'nav_goal': nav_goal}, index=[0])
        return file_name, frame
    

    def save_text_label(self, file_path, frame):
        '''save as csv'''
        if len(frame) != 0: 
                frame.to_csv(file_path, mode='a+', index=False, header=False)

    def batch_convert(self, info_df):
        '''batch process and save info'''
        tqdm.pandas(desc='Concatenating original info to csv')
        text_label_name_len_series = info_df.progress_apply(
            self.convert_info, axis=1)
        for i in tqdm(range(len(text_label_name_len_series)), desc='Saving text information'):
            filename, frame = text_label_name_len_series[i]
            self.save_text_label(self.csv_path, frame)


def get_available_car_data(start_date, end_date):

    sql = f"""
    SELECT 
    date_period    
    ,car_id
    ,begin_timestamp as start_ts
    ,end_timestamp as end_ts
    FROM task_control_info
    WHERE task_name='ego_vehicle_feature' AND task_status=2 and date_period >= '{start_date}' and  date_period <='{end_date}'
    ORDER BY date_period DESC 
    """
    postgres_helper = get_postgres_helper(database='common')
    car_data_df = postgres_helper.read_db_to_df(sql)
    return car_data_df


def get_sql_str(car_id, start_ts, end_ts):
    scene_name = '左变道'
    nav_goal = 1
    # 1：左变道，2：右变道，3：直行（巡航），4：左掉头，5：路口左转，6：路口右转
    sql = f"""
        SELECT
        ego.begin_ts
        ,ego.end_ts
        ,ego.car_id
        ,ego.curr_ts
        ,scene.event_process
        ,ego.steer_angle
        ,ego.vel_x
        ,ego.vel_y
        ,ego.accel_x
        ,ego.accel_y
        ,ego.euler_yaw
        ,COALESCE(lane_line.line_info, '[]')  lane_info
        ,COALESCE(obs.obs_info, '[]')  obs_info
        ,{nav_goal} as nav_goal
    FROM
    (
        SELECT    -- 场景
            s_2.begin_ts
            ,s_2.end_ts
            ,s_2.car_id
            ,s_2.event_process         -- 场景事件（1：事件前，2：事件中，3：事件后）
        FROM
        (
            SELECT
                s_1.begin_ts
                ,s_1.end_ts
                ,s_1.car_id
                ,s_1.before_event_process
                ,s_1.after_event_process
                ,(
                    CASE
                        WHEN s_1.before_event_process>0 AND s_1.after_event_process=0 THEN 1   -- 事件前
                        WHEN s_1.before_event_process>0 AND s_1.after_event_process>0 THEN 2   -- 事件中
                        WHEN s_1.before_event_process=0 AND s_1.after_event_process>0 THEN 3   -- 事件后
                        ELSE 0
                    END
                ) event_process
            FROM
            ( 
                SELECT
                    begin_ts
                    ,end_ts
                    ,car_id
                    ,scene_name
                    ,MAX(
                        CASE
                            WHEN POSITION('{scene_name}' IN scene_name)>0 THEN 1       -- 变道中
                            ELSE 0
                        END
                    ) OVER(PARTITION BY car_id ORDER BY curr_ts ROWS BETWEEN CURRENT ROW AND 50 FOLLOWING) before_event_process
                    ,MAX(
                        CASE
                            WHEN POSITION('{scene_name}' IN scene_name)>0 THEN 1       -- 变道中
                            ELSE 0
                        END
                    ) OVER(PARTITION BY car_id ORDER BY curr_ts ROWS BETWEEN 50 PRECEDING AND CURRENT ROW) after_event_process
                FROM ps_dynamic_scene_pool
                WHERE car_id = '{car_id}' AND curr_ts>='{start_ts}' AND curr_ts<='{end_ts}'
            ) s_1
        ) s_2
        WHERE s_2.event_process>0
    ) scene
    LEFT JOIN
    (
        SELECT
            begin_ts
            ,end_ts
            ,car_id
            ,curr_ts
            ,steer_angle
            ,vel_x
            ,vel_y
            ,accel_x
            ,accel_y
            ,euler_yaw
        FROM ps_ego_vehicle_feature
        WHERE car_id = '{car_id}' AND curr_ts>='{start_ts}' AND curr_ts<='{end_ts}'
    ) ego
    ON 
    scene.begin_ts = ego.begin_ts AND scene.end_ts = ego.end_ts AND scene.car_id = ego.car_id 
    LEFT JOIN
    (
        SELECT 
            begin_ts
            ,end_ts
            ,car_id
            ,'[' || string_agg(cast(line_a.line_info as text), ',') || ']' line_info
        FROM
        (
            SELECT
                begin_ts
                ,end_ts
                ,car_id
                ,json_build_object(
                    'line_id', line_id
                    ,'line_color', line_color
                    ,'line_type', line_type
                    ,'line_pos', line_pos
                    ,'coef_a', coef_a
                    ,'coef_b', coef_b
                    ,'coef_c', coef_c
                    ,'coef_d', coef_d
                    ,'distance', distance_center
                    ,'curvature_radius', curvature_radius
                    ,'line_start_x', line_start_x
                    ,'line_end_x', line_end_x
                ) line_info
            FROM ps_line_feature
            WHERE car_id = '{car_id}' AND curr_ts>='{start_ts}' AND curr_ts<='{end_ts}'
        ) line_a
        GROUP BY
            line_a.begin_ts
            ,line_a.end_ts
            ,line_a.car_id
    ) lane_line  
    ON scene.begin_ts = lane_line.begin_ts AND scene.end_ts = lane_line.end_ts AND scene.car_id = lane_line.car_id
    LEFT JOIN
    (
        SELECT 
            obs_a.begin_ts
            ,obs_a.end_ts
            ,obs_a.car_id
            ,'[' || string_agg(cast(obs_a.obs_info as text), ',') || ']' obs_info
        FROM
        (
            SELECT
                begin_ts
                ,end_ts
                ,car_id
                ,json_build_object(
                    'obs_id', obs_id
                    ,'obs_type', obs_type
                    ,'geo_x', cast(obs_geo::json ->> 'geo_x' AS DECIMAL(24,10))
                    ,'geo_y', cast(obs_geo::json ->> 'geo_y' AS DECIMAL(24,10))
                    ,'geo_z', cast(obs_geo::json ->> 'geo_z' AS DECIMAL(24,10))
                    ,'distance', distance
                    ,'pos_x', pos_x
                    ,'pos_y', pos_y
                    ,'vel_x', vel_x
                    ,'vel_y', vel_y
                    ,'accel_x', accel_x
                    ,'accel_y', accel_y
                    ,'theta_angle', theta_angle
                )  obs_info
            FROM ps_obstacle_feature
            WHERE car_id = '{car_id}' AND curr_ts>='{start_ts}' AND curr_ts<='{end_ts}'
                AND pos_x>-20 AND pos_x<80 AND abs(pos_y)<13 AND (abs(pos_x)+abs(pos_y))>1
        ) obs_a
        GROUP BY
            obs_a.begin_ts
            ,obs_a.end_ts
            ,obs_a.car_id
    ) obs  
    ON 
    scene.begin_ts = obs.begin_ts AND scene.end_ts = obs.end_ts AND scene.car_id = obs.car_id 
    order by scene.begin_ts
    """
    # sql = f"""SELECT
    #     ego.begin_ts
    #     ,ego.end_ts
    #     ,ego.car_id
    #     ,ego.curr_ts
    #     ,ego.steer_angle
    #     ,ego.vel_x
    #     ,ego.vel_y
    #     ,ego.accel_x
    #     ,ego.accel_y
    #     ,ego.euler_yaw
    #     ,COALESCE(lane_line.line_info, '[]')  lane_info
    #     ,COALESCE(obs.obs_info, '[]')  obs_info
    # FROM
    # ps_dynamic_scene_pool scene
    # LEFT JOIN
    # (
    #     SELECT
    #         begin_ts
    #         ,end_ts
    #         ,car_id
    #         ,curr_ts
    #         ,steer_angle
    #         ,vel_x
    #         ,vel_y
    #         ,accel_x
    #         ,accel_y
    #         ,euler_yaw
    #     FROM ps_ego_vehicle_feature
    #     WHERE car_id = '{car_id}' AND curr_ts>='{start_ts}' AND curr_ts<='{end_ts}'
    # ) ego
    # ON 
    # scene.begin_ts = ego.begin_ts AND scene.end_ts = ego.end_ts AND scene.car_id = ego.car_id 
    # LEFT JOIN
    # (
    #     SELECT 
    #         begin_ts
    #         ,end_ts
    #         ,car_id
    #         ,'[' || string_agg(cast(line_a.line_info as text), ',') || ']' line_info
    #     FROM
    #     (
    #         SELECT
    #             begin_ts
    #             ,end_ts
    #             ,car_id
    #             ,json_build_object(
    #                 'line_id', line_id
    #                 ,'line_color', line_color
    #                 ,'line_type', line_type
    #                 ,'line_pos', line_pos
    #                 ,'coef_a', coef_a
    #                 ,'coef_b', coef_b
    #                 ,'coef_c', coef_c
    #                 ,'coef_d', coef_d
    #                 ,'distance', distance_center
    #                 ,'curvature_radius', curvature_radius
    #                 ,'line_start_x', line_start_x
    #                 ,'line_end_x', line_end_x
    #             ) line_info
    #         FROM ps_line_feature
    #         WHERE car_id = '{car_id}' AND curr_ts>='{start_ts}' AND curr_ts<='{end_ts}'
    #     ) line_a
    #     GROUP BY
    #         line_a.begin_ts
    #         ,line_a.end_ts
    #         ,line_a.car_id
    # ) lane_line  
    # ON scene.begin_ts = lane_line.begin_ts AND scene.end_ts = lane_line.end_ts AND scene.car_id = lane_line.car_id
    # LEFT JOIN
    # (
    #     SELECT 
    #         obs_a.begin_ts
    #         ,obs_a.end_ts
    #         ,obs_a.car_id
    #         ,'[' || string_agg(cast(obs_a.obs_info as text), ',') || ']' obs_info
    #     FROM
    #     (
    #         SELECT
    #             begin_ts
    #             ,end_ts
    #             ,car_id
    #             ,json_build_object(
    #                 'obs_id', obs_id
    #                 ,'obs_type', obs_type
    #                 ,'geo_x', cast(obs_geo::json ->> 'geo_x' AS DECIMAL(24,10))
    #                 ,'geo_y', cast(obs_geo::json ->> 'geo_y' AS DECIMAL(24,10))
    #                 ,'geo_z', cast(obs_geo::json ->> 'geo_z' AS DECIMAL(24,10))
    #                 ,'distance', distance
    #                 ,'pos_x', pos_x
    #                 ,'pos_y', pos_y
    #                 ,'vel_x', vel_x
    #                 ,'vel_y', vel_y
    #                 ,'accel_x', accel_x
    #                 ,'accel_y', accel_y
    #                 ,'theta_angle', theta_angle
    #             )  obs_info
    #         FROM ps_obstacle_feature
    #         WHERE car_id = '{car_id}' AND curr_ts>='{start_ts}' AND curr_ts<='{end_ts}'
    #             AND pos_x>-20 AND pos_x<80 AND abs(pos_y)<13 AND (abs(pos_x)+abs(pos_y))>1
    #     ) obs_a
    #     GROUP BY
    #         obs_a.begin_ts
    #         ,obs_a.end_ts
    #         ,obs_a.car_id
    # ) obs  
    # ON 
    # scene.begin_ts = obs.begin_ts AND scene.end_ts = obs.end_ts AND scene.car_id = obs.car_id 
    
    # WHERE scene.car_id = '{car_id}' AND scene.curr_ts>='{start_ts}' AND scene.curr_ts<='{end_ts}' AND
    # (POSITION('左变道' IN scene_name)>0 OR POSITION('右变道' IN scene_name)>0 OR POSITION('旁车切入' IN scene_name)>0)
    # order by scene.begin_ts
    # """
    return sql

    #POSITION('左变道' IN scene_name)>0  OR POSITION('右变道' IN scene_name)>0 OR POSITION('跟车' IN scene_name)>0 OR  POSITION('巡航' IN scene_name)>0 OR  POSITION('旁车切入' IN scene_name)>0  OR POSITION('横向切入' IN scene_name)>0 


def get_data_and_save(sql=None):
    postgres_helper = get_postgres_helper(database='pleasures')
    info_df = postgres_helper.read_db_to_df(sql)

    info_processor = InfoProcessor(
        csv_path=args.csv_path,
        centroid_path=args.centroid_path,
    )
    info_processor.batch_convert(info_df)


def generate_csv_title():
    csv_dir = args.csv_path[0:args.csv_path.rindex('/')]
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    # 每次运行先保证源文件删除
    
    if os.path.exists(args.csv_path):
        os.remove(args.csv_path)

    with open(args.csv_path, 'w') as csvf:
        #新建csv表头列表
        fieldnames = ['car_id', 'ts', 'vel_x', 'acc_x', 'steer_angle',
                      'head_yaw_ang', 'obs_info', 'lane_info', 'centroid', 'file_name', 'waypoint_x', 'waypoint_y','nav_goal']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        #写入表头
        writer.writeheader()

def argparser():
    parser = argparse.ArgumentParser(description='Parameters of setting')
    parser.add_argument('--centroid_path', default='/data/wangyue/model_one/config_service/centroid.json', type=str,
                        help='img path to save img_feature')
    parser.add_argument('--csv_path', default='/data3/data_haomo/m1/csv/1212/pretrain_left_change_seq2seq_02.csv', type=str,
                        help='text path to save text')
    parser.add_argument('--is_auto', default=True,
                        type=bool, help='where True is auto running data')
    parser.add_argument('--car_id', default='HP-30-V71-R-208',
                        type=str, help='car id ')
    parser.add_argument('--start_ts', default=1667971352100000,
                        type=int, help='timestampe from start time')
    parser.add_argument('--end_ts', default=1667971365100000,
                        type=int, help='timestampe from end time')
    parser.add_argument('--start_date', default='2022-04-06',
                        type=str, help='timestampe from start date')
    parser.add_argument('--end_date', default='2022-12-16',
                        type=str, help='timestampe from end date')
    parser.add_argument('--nums', default= 4400 , type=int)
    parser.add_argument('--scene_name', default='',type=str, help='scene name')
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = argparser()
    generate_csv_title()
    car_data_df = get_available_car_data(args.start_date, args.end_date)
    if (car_data_df.shape[0] > 0 and args.is_auto):
        for idx, item in car_data_df.iterrows():
            car_id = item['car_id']
            start_ts = item['start_ts']
            end_ts = item['end_ts']

            sql = get_sql_str(car_id, start_ts, end_ts)
            get_data_and_save(sql)

    else:
        sql = get_sql_str(args.car_id, args.start_ts, args.end_ts)
        get_data_and_save(sql)
