'''
Author: Jiang Hang, Zhuliangqin,wangyue
Date: 2022-06-23 19:13:00
LastEditors: aiwenjie aiwenjie20@outlook.com
LastEditTime: 2022-10-19 14:29:52
FilePath: /scene_recognition/configs/config_args.py
Description: parse config file.

Copyright (c) 2022 by Haomo, All Rights Reserved.
'''
#######################################################################################################
import argparse
 

def argparser():

    parser = argparse.ArgumentParser("Train and evaluate config", add_help=False)

    # directory for yaml setting file
    parser.add_argument('--two_stream_yaml', default='/data/zhuliangqin/scene_recognition/configs/mmnet.yaml', type=str, help='Choosing two stream yaml file.')
    parser.add_argument('--one_stream_yaml', default='/data/zhuliangqin/scene_recognition/configs/one_stream.yaml', type=str, help='Choosing space-time yaml file.')
    parser.add_argument('--single_image_Yaml', default='/data/zhuliangqin/scene_recognition/configs/mmstnet.yaml', type=str, help='Choosing space-time yaml file.')


    # directory for text
    parser.add_argument('--text_path', type=str, default='/data3/data_haomo/scene_recognition/txt/bd_0930', help='concatenated text info.')
    parser.add_argument('--eval_text_path', type=str, default='/data3/data_haomo/scene_recognition/txt/eval_bd_0930', help='concatenated text info.')
    
    # directory for text token
    parser.add_argument('--text_token_path', type=str, default='/data3/data_haomo/scene_recognition/txt_token/bd_0930', help='text token after tokenizing.')
    parser.add_argument('--eval_text_token_path', type=str, default='/data3/data_haomo/scene_recognition/txt_token/eval_bd_0930', help='text token V2 after tokenizing.')
    
    # directory for image
    parser.add_argument('--origin_img_path', type=str, default='/data3/data_haomo/scene_recognition/img/bd_0930/eval_all_wide_camera_img/', help='Original image downloaded from database.')
    parser.add_argument('--matched_img_path', type=str, default='/data3/data_haomo/scene_recognition/img/bd_0930/matched_img/', help='matched the timestamp image.')
    parser.add_argument('--train_img_path', type=str, default='/data3/data_haomo/scene_recognition/img/bd_0930/train_img/', help='combined img for train.')
    parser.add_argument('--eval_img_path', type=str, default='/data3/data_haomo/scene_recognition/img/bd_0930/eval_img/', help='img for evaluate.')

    # directory for bev path   
    parser.add_argument('--bev_path', type=str, default='/data3/data_haomo/scene_recognition/bev/bd_0930', help='bev for train.')
    # parser.add_argument('--train_bev_path', type=str, default='/data3/data_haomo/scene_recognition/txt/bd_0925', help='bev for train.')
    parser.add_argument('--eval_bev_path', type=str, default='/data3/data_haomo/scene_recognition/bev/eval_bd_0930', help='bev for evaluate.')
    
    # directory for pretrained model or tokenizer
    parser.add_argument('--pretrained_model_path', type=str, default='/data/zhuliangqin/scene_recognition/pretrained_model', help='path which saves pretrained model.')
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()

    return args
