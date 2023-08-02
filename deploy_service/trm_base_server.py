'''
Author: wangyue
Date: 2023-01-08 18:06:08
LastEditTime: 2023-01-08 13:49:59
Description:
# 模型推理服务
'''
# encoding utf-8
import json
import os
import traceback
from bottle import route, run, request

import logging
import logging.handlers


project_name = 'ptrt_base_server'
formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d - %(process)d - %(levelname)s - %(pathname)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')


log_base_path = '/data3/data_haomo/m1/logs/'

full_logger_fh = log_base_path + project_name + '/' + project_name + '_full.log'
full_rh = logging.handlers.TimedRotatingFileHandler(full_logger_fh, 'D')
full_rh.setFormatter(formatter)
full_logger = logging.getLogger('full_logger')
full_logger.addHandler(full_rh)
full_logger.setLevel(logging.INFO)

request_logger_fh = log_base_path + project_name + \
    '/' + project_name + '_request.log'
request_rh = logging.handlers.TimedRotatingFileHandler(request_logger_fh, 'D')
request_rh.setFormatter(formatter)
request_logger = logging.getLogger('request_logger')
request_logger.addHandler(request_rh)
request_logger.setLevel(logging.INFO)

error_logger_fh = log_base_path + project_name + '/' + project_name + '_error.log'
error_rh = logging.handlers.TimedRotatingFileHandler(error_logger_fh, 'D')
error_rh.setFormatter(formatter)
error_logger = logging.getLogger('error_logger')
error_logger.addHandler(error_rh)
error_logger.setLevel(logging.INFO)


@route('/', methods=['GET', 'POST'])
def base():
    return project_name + " Application"


@route('/trajs_predict', method='GET')
def trajs_predict():
    try:
        result = {
            'code': 0,
            'message': '成功'
        }

        nav_goal = request.query.get('nav_goal','')
        print(f'nav_goal = {nav_goal}')
        
        car_id = request.query.get('car_id','')
        print(f'car_id = {car_id}')

        begin_ts = request.query.get('begin_ts','')
        print(f'begin_ts = {begin_ts}')

        end_ts = request.query.get('end_ts','')
        print(f'end_ts = {end_ts}')

        # 调用推理模块
        result['trajs'] = [(1,2),(2,1),(1,1)]

    except Exception:
        error_logger.error('how_to_answer:{}'.format(traceback.format_exc()))
        result['code'] = 2
        result['message'] = '请求出错'
        result['traceback'] = traceback.format_exc()

    full_logger.info('pid:{}, trajs_predict:{}'.format(
        os.getpid(), json.dumps(result, ensure_ascii=True)))
    return result


@route('/test', methods=['GET'])
def test():
    return "result"


if __name__ == '__main__':
    run(host="0.0.0.0", port=16888)
    print(os.getpid())
    print("Application Start Success")
