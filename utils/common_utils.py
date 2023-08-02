from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

def right_round(num, keep_n):
    if isinstance(num, float):
        num = str(num)
    return Decimal(num).quantize((Decimal('0.' + '0'*keep_n)), rounding=ROUND_HALF_UP)


def round_str(num, keep_n):
    if isinstance(num, str):
        num = float(num)
    return f'{num:.{keep_n}f}'

def round_num(num, keep_n):
    if isinstance(num, str):
        num = float(num)
    return float(f'{num:.{keep_n}f}')

# print(type(round_num(1.2001,3)))

def get_ts_info(data_path='/data3/data_haomo/m1/csv/1212/pretrain_lane_change_seq2seq_02.csv'):
    df = pd.read_csv(data_path)
    result = np.unique(df['car_id'], return_index=True)
    car_id,first_idx = result[0].reshape(-1,1),result[1].reshape(-1,1)
    result = np.concatenate([car_id,first_idx],axis=1)
    result = result[np.lexsort((result[:,1],))]
    last_idx = np.row_stack((result[:,1].reshape(-1,1),np.array(len(df)).reshape(1,1)))[1:] - 1
    result = np.concatenate([result,last_idx],axis=1)

    info = []
    for row in result:
        cache = [row[0],df['ts'][row[1]],df['ts'][row[2]]]
        info.append(cache)
    info = np.array(info)
    info_df = pd.DataFrame(info,columns=["car_id","begin_ts","end_ts"])
    info_df.to_csv('/data/wangyue/model_one/data_service/info.csv',index=False,na_rep=0)