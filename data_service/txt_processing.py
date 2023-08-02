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
from fs_ossfs import OSSFS
import csv
import math
import argparse
import pandas as pd
from tqdm import tqdm
import json
import os
from fileinput import filename


project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)


class InfoProcessor:
    '''process columns info'''

    def __init__(self, csv_path):
        '''
        text_path: the path which saves chinese text.
        '''
        self.csv_path = csv_path

    def convert_info(self, columns_info):
        ''''convert columns info to text info
        @columns_info: dataframe data's every column information, as a parameter for pandas.apply.
        '''
        ego_info = eval(columns_info['ego_info'])
        fps_sin_dis_yaw = math.sin(ego_info['euler_yaw'])
        fps_cos_dis_yaw = math.cos(ego_info['euler_yaw'])
        dis_x = ego_info['vel_x']*0.1*fps_cos_dis_yaw
        dis_y = ego_info['vel_x']*0.1*fps_sin_dis_yaw
        ego_info['dis_x'] = dis_x
        ego_info['dis_y'] = dis_y

        file_name = columns_info['car_id'] + \
            '_' + str(columns_info['curr_ts'])
        frame = pd.DataFrame({'car_id': columns_info['car_id'],
                              'ts': columns_info['curr_ts'],
                              'road_event_id': columns_info['road_event_id'],
                              'road_tag': columns_info['road_tag'],
                              'obs_event_id': columns_info['obs_event_id'],
                              'obs_tag': columns_info['obs_tag'],
                              'ego_motion': columns_info['ego_motion'],
                              'navigation_intent': columns_info['navigation_intent'],
                              'accs_interv_status': columns_info['accs_interv_status'],
                              'noh_interv_status': columns_info['noh_interv_status'],
                              'ego_info': columns_info['ego_info'],
                              'obs_info': columns_info['obs_info'],
                              'lane_info': columns_info['line_info'],
                              'file_name': file_name,
                              'waypoint_x': dis_x,
                              'waypoint_y': dis_y}, index=[0])
        return file_name, frame

    def save_text_label(self, file_path, frame):
        '''save as csv'''
        if len(frame) != 0:
            frame.to_csv(file_path, mode='a+', index=False, header=False)

    def batch_convert(self, info_df):
        '''batch process and ssave info'''
        tqdm.pandas(desc='Concatenating original info to csv')
        text_label_name_len_series = info_df.progress_apply(
            self.convert_info, axis=1)
        for i in tqdm(range(len(text_label_name_len_series)), desc='saving text information'):
            filename, frame = text_label_name_len_series[i]
            self.save_text_label(self.csv_path, frame)


def get_available_data(start_date, end_date):

    sql = f"""
    SELECT 
    date_period
    ,project_name
    ,car_id
    ,CAST((pay_data_address::JSON ->> 'db_name') AS VARCHAR(64)) db_name
    ,CAST((pay_data_address::JSON ->> 'table_name') AS VARCHAR(512)) table_name
    FROM application_record_info
    WHERE app_org='alg_cognize_big_model' AND app_name='all_scene'
    ORDER BY date_period,car_id ASC  
    """
    postgres_helper = get_postgres_helper(database='common')
    record_info_df = postgres_helper.read_db_to_df(sql)
    return record_info_df


def get_oss_file_to_df(db_name, table_name):
    ossfs = OSSFS(
        bucket_name=db_name,
        aws_access_key_id='LTAI5tNngauABWNaAZUbUALS',
        aws_secret_access_key='HEKrKUBzaZKCN0An0TdJIRxCS9mdoz',
        endpoint_url='https://oss-cn-beijing-internal.aliyuncs.com'
    )
    traj_data_df = pd.read_parquet(
        path=ossfs.geturl(table_name), engine='pyarrow')
    return traj_data_df


def process_data_and_save(traj_data_df=None):
    info_processor = InfoProcessor(
        csv_path=args.csv_path,
    )
    info_processor.batch_convert(traj_data_df)


def generate_csv_title():
    csv_dir = args.csv_path[0:args.csv_path.rindex('/')]
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    with open(args.csv_path, 'w') as csvf:
        #新建csv表头列表
        fieldnames = ['car_id', 'ts', 'road_event_id', 'road_tag', 'obs_event_id',
                      'obs_tag', 'ego_motion', 'navigation_intent', 'accs_interv_status',
                      'noh_interv_status','ego_info','obs_info','lane_info', 'file_name', 
                      'waypoint_x', 'waypoint_y']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        #写入表头
        writer.writeheader()


def argparser():
    parser = argparse.ArgumentParser(description='Parameters of setting')
    parser.add_argument('--csv_path', default='/data3/data_haomo/m1/csv/0131/pretrain_seq2seq_01.csv', type=str,
                        help='text path to save text')
    parser.add_argument('--is_auto', default=False,
                        type=bool, help='where True is auto running data')
    parser.add_argument('--car_id', default='HP-30-V71-R-205',
                        type=str, help='car id ')
    parser.add_argument('--start_ts', default=1668733393000000,
                        type=int, help='timestampe from start time')
    parser.add_argument('--end_ts', default=1668733403000000,
                        type=int, help='timestampe from end time')
    parser.add_argument('--start_date', default='2022-10-15',
                        type=str, help='timestampe from start date')
    parser.add_argument('--end_date', default='2022-11-10',
                        type=str, help='timestampe from end date')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()
    record_info_df = get_available_data(args.start_date, args.end_date)

    if (record_info_df.shape[0] > 0):
        generate_csv_title()
        for idx, item in record_info_df.iterrows():
            date_period = item['date_period']
            car_id = item['car_id']
            db_name = item['db_name']
            table_name = item['table_name']

            traj_data_df = get_oss_file_to_df(db_name, table_name)
            process_data_and_save(traj_data_df)
