'''
对多个csv文件合并操作
'''
import sys
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


def argparser():
    parser = argparse.ArgumentParser(description='Parameters of setting')
    parser.add_argument('--input_csv_path', default='/data3/data_haomo/m1/csv/1212/', type=str,
                        help='input csv path')
    parser.add_argument('--input_csv_files', default='pretrain_left_change_seq2seq_02.csv,pretrain_right_change_seq2seq_02.csv', type=str,
    help='input csv files')
    parser.add_argument('--output_csv_path', default='/data3/data_haomo/m1/csv/1212/pretrain_lane_change_seq2seq_02.csv', type=str,
                        help='output csv path')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = argparser()
    input_csv_files = args.input_csv_files.split(',')
    merge_file = pd.DataFrame()
    # 遍历所有待合并文件，并做合并操作
    for input_csv_file in input_csv_files:
        file = pd.read_csv(os.path.join(args.input_csv_path,input_csv_file))

        if merge_file.empty:
            merge_file = file
        else :
            # merge_file = pd.concat([merge_file,file])
            merge_file =  merge_file.append(file)
            
    
    # 按照ts和carid升序排序
    merge_file.sort_values(by=['car_id','ts'],ascending=[True,True])

    # 保存csv数据
    merge_file.to_csv(args.output_csv_path, mode='w', index=False, header=True)


    
    
