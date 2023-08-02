"""
@Project: model_one
@File: img_processor.py
@Author: wangyue
@Date: 2022/6/25 15:05
@LastEditTime: 2022/6/27 16:20
@Description: image preprocessing module, including frame extraction, scaling, etc.

@License: Copyright (c) 2022 by Haomo, All Rights Reserved.
"""
import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import cv2
import logging
import numpy as np
from tqdm import tqdm
from glob import glob
from functools import partial
from multiprocessing import Pool



# from ...configs.config_args import argparser


class ImgProcessor(object):
    """
    图像数据预处理器
    """
    """
    get_img, get_history_img, get_current_img在图片数量巨大时(超过10万张)效率低下
    """

    def __init__(self,
                 # config,
                 is_same_ratio=True,
                 is_combined=False):
        """
        初始化图像数据预处理模块
        @param is_same_ratio: 是否等比例缩放
        @param is_combined: 是否拼接图像
        """
        # self.config = config
        self.is_same_ratio = is_same_ratio
        self.is_combined = is_combined

        logging.info('Image processor is initialized')

    def combine_img(self,
                    pers_imgs: list,
                    add_black=False,
                    black_width=5):
        """
        拼接五个视角的图片(针对原图640*360的特定拼接方法)
        @param pers_imgs: 存放五个视角图像的List
        @param add_black: 是否填充黑边
        @param black_width: 黑边的宽度
        @return: 拼接后的图像
        """
        if add_black:
            h_black = np.zeros((black_width, 640, 3), dtype=np.uint8)
            v1_black = np.zeros((270, black_width, 3), dtype=np.uint8)
            v2_black = np.zeros((180, black_width, 3), dtype=np.uint8)
            
            if pers_imgs[0] is None or pers_imgs[1] is None or pers_imgs[2] is None or pers_imgs[3] is None or pers_imgs[4] is None  :
                return None

            
            # 将左前和右前的图像缩放为(480, 270)再裁剪掉与前向图片相同的部分尺寸变为(320, 270)
            lf_img, rf_img = cv2.resize(pers_imgs[4], (480, 270))[0:270, 0:320], cv2.resize(pers_imgs[1], (480, 270))[
                                                                                 0:270, 160 + black_width:480]
            # 将左后和右后的图像缩放为(320, 180)
            lr_img, rr_img = cv2.resize(pers_imgs[3], (320, 180)), cv2.resize(pers_imgs[2], (320, 180))[0:180,
                                                                   black_width:320]
            # 将左前和右前的图像水平方向拼接尺寸变为(640, 270)
            lf_rf_img = cv2.hconcat([lf_img, v1_black, rf_img])
            # 将左后和右后的图像水平方向拼接尺寸变为(640, 180)
            lr_rr_img = cv2.hconcat([lr_img, v2_black, rr_img])
            # 将5个视角图像拼接尺寸变为(640, 810)
            combined_img = cv2.vconcat([lf_rf_img, h_black, pers_imgs[0], h_black, lr_rr_img])

            combined_img = cv2.resize(combined_img, (380, 380))
        else:
            # 将左前和右前的图像缩放为(480, 270)再裁剪掉与前向图片相同的部分尺寸变为(320, 270)
            lf_img, rf_img = cv2.resize(pers_imgs[4], (480, 270))[0:270, 0:320], cv2.resize(pers_imgs[1], (480, 270))[
                                                                                 0:270, 160:480]
            # 将左后和右后的图像缩放为(320, 180)
            lr_img, rr_img = cv2.resize(pers_imgs[3], (320, 180)), cv2.resize(pers_imgs[2], (320, 180))
            # 将左前和右前的图像水平方向拼接尺寸变为(640, 270)
            lf_rf_img = cv2.hconcat([lf_img, rf_img])
            # 将左后和右后的图像水平方向拼接尺寸变为(640, 180)
            lr_rr_img = cv2.hconcat([lr_img, rr_img])
            # 将5个视角图像拼接尺寸变为(640, 810)
            combined_img = cv2.vconcat([lf_rf_img, pers_imgs[0], lr_rr_img])
        return combined_img

    def combine_imgs(self,
                     add_black,
                     black_width,
                     combine_path: str,
                     image_list) -> None:
        """
        拼接多视角的图片集合
        @param add_black: 是否填充黑边
        @param black_width: 填充黑边的宽度
        @param image_list: 提取后的图像数据路径
        @param combine_path: 拼接后的图像数据路径
        @return: None
        """
        # # 读取文件路径
        # files = os.listdir(image_list)
        # 存放六个视角图像
        pers_imgs = []

        # 遍历文件夹下文件
        for file in tqdm(image_list, desc='combine images'):

            # 读取图像
            img = cv2.imread(file)
            pers_imgs.append(img)

            if len(pers_imgs) == 5:
                combined_img = self.combine_img(pers_imgs[:], add_black, black_width)

                # 保存拼接后的图片
                cv2.imwrite(combine_path + os.path.basename(file)[0:22] + '.jpg', combined_img)
                # 清除六个视角图像
                pers_imgs.clear()

    def resize_img(self,
                   input_img: np,
                   height: int,
                   width: int) -> np:
        """
        修改单个图像的尺寸
        @param input_img: 输入图像
        @param height: 生成图像的高
        @param width: 生成图像的宽
        @return: 修改尺寸后的图像
        """
        # 无需等比例缩放
        if not self.is_same_ratio:
            resize_img = cv2.resize(input_img, (width, height))
            return resize_img

        # 获取长宽更短的一边
        min_side = min(height, width)

        # 获得原图的高宽尺寸
        origin_size = input_img.shape
        origin_height, origin_width = origin_size[0], origin_size[1]

        # 计算中间图的尺寸, 中间图与最终图的短边仅差至多1个像素点
        scale = max(origin_height, origin_width) / float(min_side)
        middle_height, middle_width = int(origin_height / scale), int(origin_width / scale)

        # 先将原图尺寸修改到中间图
        middle_img = cv2.resize(input_img, (middle_width, middle_height))

        # 计算图像长宽两侧大致需要填充的像素点个数
        padding_width = (width - middle_width) / 2
        padding_height = (height - middle_height) / 2

        # 计算middle_height * middle_width到height * width四边需要填充的像素点
        if (width - middle_width) % 2 != 0 and (height - middle_height) % 2 == 0:
            top, bottom, left, right = padding_height, padding_height, padding_width + 1, padding_width
        elif (width - middle_width) % 2 == 0 and (height - middle_height) % 2 != 0:
            top, bottom, left, right = padding_height + 1, padding_height, padding_width, padding_width
        elif (width - middle_width) % 2 == 0 and (height - middle_height) % 2 == 0:
            top, bottom, left, right = padding_height, padding_height, padding_width, padding_width
        else:
            top, bottom, left, right = padding_height + 1, padding_height, padding_width + 1, padding_width

        # 从图像边界向上,下,左,右扩的像素数目
        resize_img = cv2.copyMakeBorder(middle_img, int(top), int(bottom), int(left), int(right),
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if height != resize_img.shape[0] or width != resize_img.shape[1]:
            print('unexpected size image')
        return resize_img

    