'''
Author: Jiang Hang, Zhuliangqin
Date: 2022-08-23 14:14:23
LastEditors: aiwenjie aiwenjie20@outlook.com
LastEditTime: 2022-11-03 16:20:47
FilePath: /scene_recognition/data_service/data_processing.py
Description: deal with original image data.

Copyright (c) 2022 by Haomo, All Rights Reserved.
'''
import sys
sys.path.append('/data3/data_haomo/project/data-service')
from production.common_method.common_method import get_picture_addr
from common.util.db.postgres_helper import get_postgres_helper

import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import requests
import argparse
import time
import os
from utils.cfg_utils import load_yaml2cfg
from builtins import print



def argparser():
    parser = argparse.ArgumentParser(description='Parameters of setting')
    parser.add_argument('--image_dir', default='/data3/data_haomo/m1/img/1115/', type=str,
                        help='save dir of images')
    parser.add_argument('--is_save_image', default=True,
                        type=bool, help='Whether or not to save image file')
    parser.add_argument('--project_name', default="icu30",
                        type=str, help='Data project name')
    parser.add_argument('--camera_name', default="front_wide_camera_record", type=str,
                        help='Download the name of the image perspective')
    parser.add_argument('--is_train', default=True, type=bool,
                        help='Whether or not  to  get train data')

    parser.add_argument('--car_id', default='HP-30-V71-R-205',
                        type=str, help='car id ')
    parser.add_argument('--start_ts', default=1665641596095000,
                        type=int, help='timestampe from start time')
    parser.add_argument('--end_ts', default=1665645275983000,
                        type=int, help='timestampe from end time')

    parser.add_argument('--start_date', default='2022-10-15',
                        type=str, help='timestampe from start date')
    parser.add_argument('--end_date', default='2022-11-10',
                        type=str, help='timestampe from end date')

    args = parser.parse_args()
    return args


args = argparser()

qur_sql_eval = """
    select
        begin_ts
        ,end_ts
        ,car_id
        ,curr_ts
    from alg_scene_recognition 
    WHERE
        scene_name IN (
            SELECT
                scene_name
            FROM
                alg_scene_recognition
            WHERE
                record_version = 3
            GROUP BY
                scene_name
            HAVING
                count(curr_ts) > 100) and  record_version = 3
    """


def __download_picture(curr_row):
    begin_ts = curr_row.begin_ts * 1000
    end_ts = curr_row.end_ts * 1000
    curr_ts = curr_row.curr_ts
    project_name = args.project_name
    car_id = curr_row.car_id
    camera_name_list = [args.camera_name]

    picture_addr = get_picture_addr(
        project_name, car_id, camera_name_list, begin_ts, end_ts)

    for images in tqdm(picture_addr):
        image = requests.get(images.get("image_url"))
        full_path = os.path.join(
            args.image_dir, car_id + '_' + str(curr_ts) + '.jpg')
        with open(full_path, 'wb') as img_fp:
            img_fp.write(image.content)
    return 0


def multi_data_process(data):
    data.apply(__download_picture, axis=1)


def get_img_info(sql=None):
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)
    # 获取CSV数据
    postgres_helper = get_postgres_helper(database='pleasures')
    car_data_df = postgres_helper.read_db_to_df(sql)

    # 下载图像
    if args.is_save_image:
        print('Image starts saving')
        start_download_img = time.time()
        # 多线程存储图片
        df_split = np.array_split(car_data_df, 32)
        with Pool(32) as pool:
            result = pool.map(multi_data_process, df_split)
        # 单线程
        # scene_recognition_car_df.apply(__download_picture, axis=1)
        end_download_img = time.time()
        print('Image saving took {} s'.format(
            end_download_img - start_download_img))


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

def  get_sql_str(car_id,start_ts,end_ts):
    sql = f"""
            SELECT begin_ts,end_ts,car_id,curr_ts FROM ps_ego_device_feature WHERE car_id = '{car_id}' AND curr_ts>='{start_ts}' AND curr_ts<='{end_ts}'
            """
    return sql


if __name__ == '__main__':

    cfg_path = '/data/wangyue/model_one/config_service/m1_config_v1.yaml'
    config = load_yaml2cfg(cfg_path)
    args = argparser()

    car_data_df = get_available_car_data(args.start_date, args.end_date)
    if car_data_df.shape[0] >0:
        for idx, item in car_data_df.iterrows():
            car_id = item['car_id']
            start_ts = item['start_ts']
            end_ts = item['end_ts']
            sql = get_sql_str(car_id,start_ts,end_ts)
            get_img_info(sql=sql)

    else:
        sql = get_sql_str(args.car_id,args.start_ts,args.end_ts)
        get_img_info(sql=sql)
