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
from flask import Flask, request
# from flask_cors import CORS

import logging
import logging.handlers


app = Flask(__name__)
# CORS(app, supports_credentials=True)

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


@app.before_request
def log_request():
    input = str(request.get_data(), encoding='utf8')
    request_url = request.url

    request_logger.info('%s - %s' % (request_url, input))


@app.after_request
def log_response(response):
    output_json = response.get_json()
    output = json.dumps(output_json, ensure_ascii=False)
    request_url = request.url

    request_logger.info('%s - %s' % (request_url, output))
    return response


@app.route('/', methods=['GET', 'POST'])
def base():
    return project_name + " Application"


# 搜索匹配句子
@app.route('/trajs_predict', methods=['GET','POST'])
def trajs_predict():
    try:
        result = {
            'code': 0,
            'message': '成功'
        }

        nav_goal = request.args.get('nav_goal','')
        print(f'nav_goal = {nav_goal}')
        
        car_id = request.args.get('car_id','')
        print(f'car_id = {car_id}')

        begin_ts = request.args.get('begin_ts','')
        print(f'begin_ts = {begin_ts}')

        end_ts = request.args.get('end_ts','')
        print(f'end_ts = {end_ts}')

        # 调用推理模块
        result['trajs'] = [(1,2),(2,1),(1,1)]

    except Exception:
        error_logger.error('how_to_answer:{}'.format(traceback.format_exc()))
        result['code'] = 2
        result['message'] = '请求出错'
        result['traceback'] = traceback.format_exc()

    app.logger.info('pid:{}, trajs_predict:{}'.format(
        os.getpid(), json.dumps(result, ensure_ascii=True)))
    return result


@app.route('/test', methods=['GET'])
def test():
    return "result"


if __name__ == '__main__':
    import os
    app.run(host="0.0.0.0", port=16888, debug=False, threaded=False)
    print(os.getpid())
    print("Application Start Success")
