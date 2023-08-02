'''
Author: wangyue
Date: 2022-11-08 18:06:08
LastEditTime: 2022-11-09 13:49:59
Description:
配置文件工具类
'''

from yacs.config import CfgNode as CN

def load_yaml2cfg(cfg_path):
    return CN.load_cfg(open(cfg_path))