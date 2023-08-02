"""
@Project: model_one
@File: get_img_info.py
@Author: zhuliangqin
@Date: 2022/7/8 12:15
@LastEditTime: 2022/7/8 12:15
@Description: get original data module.

@License: Copyright (c) 2022 by Haomo, All Rights Reserved.
"""
import os
import sys
sys.path.append('/data3/data_haomo/project/data-service')
from common.util.db.postgres_helper import get_postgres_helper
from production.common_method.common_method import get_picture_addr

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import time
import argparse
import requests
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def get_original_info(database, ego_qur_sql):
    postgres_helper = get_postgres_helper(database=database)
    info_df = postgres_helper.read_db_to_df(ego_qur_sql)
    return info_df

def argparser():
    parser = argparse.ArgumentParser(description='Parameters of setting')
    parser.add_argument('--img_dir', default='/data3/data_haomo/m1/img/1212/pretrain_lane_change', type=str,
                        help='save dir of images')
    parser.add_argument('--is_save_img', default=True, type=bool, help='Whether or not to save image file')
    parser.add_argument('--project_name', default="icu30", type=str, help='Data project name')
    parser.add_argument('--camera_name', default="front_wide_camera_record", type=str,
                        help='Download the name of the image perspective')
    parser.add_argument('--is_train', default=True, type=bool, help='Whether or not  to  get train data ')
    parser.add_argument('--car_id', default='HP-30-V71-R-211',
                        type=str, help='car id ')
    parser.add_argument('--begin_ts', default=1668991572999,
                        type=int, help='timestampe from start time')
    parser.add_argument('--end_ts', default=1668995066991,type=int, help='timestampe from end time')

    args = parser.parse_args()
    return args


args = argparser()

qur_sql_train = f"""
    SELECT
        begin_ts
        ,end_ts
        ,car_id
        ,curr_ts
    FROM ps_ego_vehicle_feature 
    WHERE
        car_id = '{args.car_id}' AND begin_ts >= {args.begin_ts} AND end_ts <= {args.end_ts}
    """


qur_sql_eval = """
    
    """


def __download_picture(curr_row):
    begin_ts = curr_row.begin_ts * 1000
    end_ts = curr_row.end_ts * 1000
    curr_ts = curr_row.curr_ts
    project_name = args.project_name
    car_id = curr_row.car_id
    camera_name_list = [args.camera_name]

    camera_name_dict = {'front_wide_camera_record': '_0',
                        'rf_wide_camera_record': '_1',
                        'rr_wide_camera_record': '_2',
                        'rear_wide_camera_record': '_3',
                        'lr_wide_camera_record': '_4',
                        'lf_wide_camera_record': '_5'}

    picture_addr = get_picture_addr(project_name, car_id, camera_name_list, begin_ts, end_ts)
    # is_picture = True if len(picture_addr) > 0 else False

    for images in tqdm(picture_addr):
        image = requests.get(images.get("image_url"))
        full_path = os.path.join(args.img_dir, (car_id) + '_' + str(curr_ts) + '.jpg')
        with open(full_path, 'wb') as img_fp:
            img_fp.write(image.content)
    return 0


def multi_data_process(data):
    data.apply(__download_picture, axis=1)


if __name__ == '__main__':

    # front_wide_camera_record
    # 前广角摄像头  2795968-30268    _0
    # rear_wide_camera_record
    # 后广角摄像头                   _3
    # lf_wide_camera_record
    # 左前中距摄像头  2788069-22369  _5
    # lr_wide_camera_record
    # 左后中距摄像头  2795710-30010  _4
    # rf_wide_camera_record
    # 右前中距摄像头  2795786-30086  _1
    # rr_wide_camera_record
    # 右后中距摄像头  2765865-165    _2

    # 检查路径，如果不存在则创建，os.makedirs是递归创建
    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)

    # 获取CSV数据
    postgres_helper = get_postgres_helper(database='pleasures')
    car_info_df = postgres_helper.read_db_to_df(qur_sql_train if args.is_train else qur_sql_eval)

    # 下载图像
    if args.is_save_img:
        print('Image starts saving')
        start_download_img = time.time()
        # 多线程存储图片
        df_split = np.array_split(car_info_df, 32)
        with Pool(32) as pool:
            result = pool.map(multi_data_process, df_split)

        # 单线程
        # scene_recognition_car_df.apply(__download_picture, axis=1)

        end_download_img = time.time()
        print('Image saving took {} s'.format(end_download_img - start_download_img))
