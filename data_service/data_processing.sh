#!/usr/bin/env bash
###
 # @Author: aiwenjie aiwenjie20@outlook.com
 # @Date: 2022-11-01 14:00:24
 # @LastEditors: aiwenjie aiwenjie20@outlook.com
 # @LastEditTime: 2022-11-07 16:12:55
 # @FilePath: /model_one/data_service/data_processing.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

cd /data/wangyue/model_one/data_service

# 下载原始图片
python img_processing.py

# 将图片转为向量
python img_encoder.py

# 找出最优的聚类个数 clusters
python /data/wangyue/model_one/model_base/calc_ch_score.py


# 图片聚类 生成中心
python /data/wangyue/model_one/model_base/minibatch_hkmeans.py --hierarchies=3 --clusters=15

# 分配 ID 
python centroid_to_img.py

# 下载文本信息
python text_processing.py