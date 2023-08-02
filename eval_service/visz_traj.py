import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
sys.path.append("..")
import cv2
import numpy as np
import math
import json
import pandas as pd
import imageio.v2 as imageio
from utils.cfg_utils import *
from tqdm import tqdm
import time

class Visualization:
    def __init__(self,cfg) -> None:
        self.cfg = cfg
    # 旋转矩阵函数
    def rotate_matrix(self, angle, rect):
        """
        args:
        angel: 绕中心旋转的角度，例：270
        rect: [x, y, w, h]，障碍物的中心坐标和宽高
        return: 旋转后顺时针顺序的四个点的坐标
        """
        anglePi = -angle * math.pi / 180.0
        cosA = math.cos(anglePi)
        sinA = math.sin(anglePi)
    
        x=rect[0]
        y=rect[1]
        width=rect[2]
        height=rect[3]
        x1 = x - 0.5 * width
        y1 = y - 0.5 * height
    
        x0 = x + 0.5 * width
        y0 = y1
    
        x2 = x1
        y2 = y + 0.5 * height
    
        x3 = x0
        y3 = y2
    
        x0n = (x0 - x) * cosA - (y0 - y) * sinA + x
        y0n = (x0 - x) * sinA + (y0 - y) * cosA + y
    
        x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
        y1n = (x1 - x) * sinA + (y1 - y) * cosA + y
    
        x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
        y2n = (x2 - x) * sinA + (y2 - y) * cosA + y
    
        x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
        y3n = (x3 - x) * sinA + (y3 - y) * cosA + y
    
        return [(x0n,y0n),(x1n,y1n),(x2n,y2n),(x3n,y3n)]

    # 只有轨迹点
    def bev_drawing_trac(self, id, frame,pred_x, pred_y, bev_path, img, gt_x,gt_y, up = 50, down = 20, left = 25, right = 25, row_rate = 5, column_rate = 5, ego_w = 2, ego_h = 5):
        # 根据输入比例尺等信息计算图像大小
        #img=img.permute(1,2,0).cpu().numpy().copy()
        image_w = int((left + right)*row_rate)
        image_h = int((up + down)*column_rate)
        # 原始图片
        # cv2.imwrite(os.path.join(bev_path,'gt.jpg'), img)
        car_w = ego_w
        car_h = ego_h
        
        # 根据主车宽高信息和比例尺计算主车坐标
        for i in range(len(pred_x)):

            ego_info = [
            [int((left + pred_y[i] - car_w/2)*row_rate), int((up - pred_x[i] - car_h/2)*column_rate)], 
            [int((left + pred_y[i] + car_w/2)*row_rate), int((up - pred_x[i] - car_h/2)*column_rate)], 
            [int((left + pred_y[i] + car_w/2)*row_rate), int((up - pred_x[i] + car_h/2)*column_rate)], 
            [int((left + pred_y[i] - car_w/2)*row_rate), int((up - pred_x[i] + car_h/2)*column_rate)]]
            rectangle = np.array(ego_info)  
            x = int((rectangle[0][0] + rectangle[1][0])/2)
            y = int((rectangle[1][1] + rectangle[2][1])/2)
            # cv2.fillConvexPoly(img, rectangle_real, (0,0,255))    # 画主车
            cv2.circle(img, (x,y), 1, (0,0,255), 1)
        
        for i in range(len(gt_x)):
            # 根据主车宽高信息和比例尺计算主车坐标
            ego_info_real = [
            [int((left + gt_y[i] - car_w/2)*row_rate), int((up - gt_x[i] - car_h/2)*column_rate)], 
            [int((left + gt_y[i] + car_w/2)*row_rate), int((up - gt_x[i] - car_h/2)*column_rate)], 
            [int((left + gt_y[i] + car_w/2)*row_rate), int((up - gt_x[i] + car_h/2)*column_rate)], 
            [int((left + gt_y[i] - car_w/2)*row_rate), int((up - gt_x[i] + car_h/2)*column_rate)]]

            rectangle_real = np.array(ego_info_real)  
            x = int((rectangle_real[0][0] + rectangle_real[1][0])/2)
            y = int((rectangle_real[1][1] + rectangle_real[2][1])/2)
            # cv2.fillConvexPoly(img, rectangle_real, (0,0,255))    # 画主车
            cv2.circle(img, (x,y), 1, (255,0,0), 1)

        cv2.imwrite(os.path.join(bev_path, f'scene_{str(id)}',f'pred_{str(frame)}.jpg'), img)

    def sum_col(self,list):
        list[0] = 0
        for i in range(1,len(list)):
            list[i] = list[i-1]+list[i]
        return list

    def bev_drawing2(self, car_id,ts, obs_info, lane_info, bev_path, up = 50, down = 20, left = 25, right = 25, row_rate = 5, column_rate = 5, color = 1, obs = False, line = False, ego_w = 2, ego_h = 5, is_rewritted = False):
        """
        Args:
        obs_info & lane_info: 每一帧的障碍物和车道线。
        obs_info: 障碍物的信息;为了同读取的数据相对应，格式为list. 使用时用json进行解析.
        lane_info: 车道线的信息，格式为list.
        bev_path: 保存bev图的路径
        ts: 时间戳，用来给BEV图命名
        up: 图像上边缘距离主车的距离（单位：m）
        down: 图像下边缘距离主车的距离（单位：m）
        left: 图像左边缘距离主车的距离（单位：m）
        right: 图像右边缘距离主车的距离（单位：m）
        row_rate:横向比例尺
        column_rate:纵向比例尺
        """

        # 根据输入比例尺等信息计算图像大小
        image_w = int((left + right)*row_rate)
        image_h = int((up + down)*column_rate)
        if is_rewritted == True:
            if not os.path.exists(bev_path+"/"+str(ts)+".jpg"):
                img = np.zeros((image_h, image_w, 3), np.uint8) + 255
            else:
                img = cv2.imread(bev_path+"/"+str(ts)+".jpg")
        else: img = np.zeros((image_h, image_w, 3), np.uint8) + 255

        # 障碍物和车道线种类颜色等设置
        obs_kind = ['car', 'truck', 'other']
        obs_color = [(0, 255, 0), (225, 105, 65), (0, 165, 255)]
 
        obs_kind_color = dict(zip(obs_kind, obs_color))

        line_kind = ['single_solid', 'road_edge', 'single_dashed', 'other']
        line_color = [(0,0,0),(0,165,255),(0,255,0),(0,0,255)]
        # line_color = [(255, 255, 255), (0, 255, 255), (18, 153, 255), (0, 69, 255)]
        line_degree = [1,2,1,1]
        line_kind_color = dict(zip(line_kind, line_color))
        line_kind_degree = dict(zip(line_kind, line_degree))
        
        try:
            # lane_info
            if line == True:
                if len(lane_info) > 0:
                    for lane_items in lane_info:
                        lane_items=json.loads(lane_items)
                        for lane_item in lane_items:
                            # 调用车道线信息画图
                            line = []
                    
                            line_type = lane_item['line_type']
                            a = float(lane_item['coef_a'])
                            b = float(lane_item['coef_b'])
                            c = float(lane_item['coef_c'])
                            d = float(lane_item['coef_d'])
                            x_start = float(lane_item['line_start_x'])
                            x_end = float(lane_item['line_end_x'])

                            # 根据车道线起始终止以及参数得到多个点，并进行坐标转换和比例尺转换
                            if x_start > x_end:
                                max_x = x_start
                                min_x = x_end
                            else:
                                min_x = x_start
                                max_x = x_end
                            for x in np.linspace(min_x, max_x, 250):
                                y = a*x**3 + b*x**2 + c*x + d    
                                x1 = round((left + (-y))*row_rate - 1)
                                y1 = round((up - x)*column_rate - 1)
                                line.append([x1,y1])

                            line = np.array(line)
                            line = line[line[:,1].argsort()]
                    
                            pts = np.array(line)
                            pts = pts.reshape((-1, 1, 2))

                            # 根据车道线类型调用设定好的车道线颜色、线型字典，获得相应参数
                            if line_type not in line_kind:
                                line_type = "other"
                            img_line_color = line_kind_color[line_type]
                            img_line_degree = line_kind_degree[line_type]
                            cv2.polylines(img, [pts], False, img_line_color, thickness = img_line_degree)         # 画车道线
                
            # obs_info
            if obs == True:           
                if len(obs_info) > 0:
                    for obs_items in obs_info:
                        obs_items=json.loads(obs_items)
                        for obs_item in obs_items:
                            # 调用障碍物信息画图
                            obs_type = obs_item["obs_type"]
                            y = float(obs_item["pos_x"])  # y=x，根据坐标的标准不同将x和y转化为对应的情形
                            x = -(float(obs_item["pos_y"]))  # x=-y
                            w = float(obs_item["geo_y"])
                            h = float(obs_item["geo_x"])
                        
                            # 调用旋转矩阵函数转换出障碍物位置
                            theta = float(obs_item["theta_angle"])
                            theta = 180*theta/math.pi
                            obs = self.rotate_matrix(theta, [x, y, w, h])  
                        
                            obs = np.array([[obs[0][0], obs[0][1]], [obs[1][0], obs[1][1]], [obs[2][0], obs[2][1]], [obs[3][0], obs[3][1]]])

                            # 对障碍物坐标进行比例尺转化，并且取整
                            for i in range(4):
                                for j in range(2):
                                    if j == 1:
                                        obs[i][j] = round((up - obs[i][j])*column_rate - 1)
                                    if j == 0:
                                        obs[i][j] = round((left + obs[i][j])*row_rate - 1)

                            rectangle = np.array([[int(obs[0][0]), int(obs[0][1])], [int(obs[1][0]), int(obs[1][1])], [int(obs[2][0]), int(obs[2][1])], [int(obs[3][0]), int(obs[3][1])]])  # 画障碍物

                            # # 为了将多帧数据上的障碍物都用颜色区分出来，根据输入的数乘上100000，转化为RGB颜色，得到颜色参数
                            # img_obs_color = getRGBfromI(color*10000000)
                            
                            # 根据障碍物类型调用设定好的车型颜色字典，获得颜色参数
                            if obs_type not in obs_kind:
                                obs_type = "other"
                            img_obs_color = obs_kind_color[obs_type]
                            # 绘制障碍物
                            cv2.fillConvexPoly(img, rectangle, img_obs_color, lineType = 32)   
                            
            # ego_info
            car_w = ego_w
            car_h = ego_h
            
            # 根据主车宽高信息和比例尺计算主车坐标
            ego_info = [
            [int((left - car_w/2)*row_rate), int((up - car_h/2)*column_rate)], 
            [int((left + car_w/2)*row_rate), int((up - car_h/2)*column_rate)], 
            [int((left + car_w/2)*row_rate), int((up + car_h/2)*column_rate)], 
            [int((left - car_w/2)*row_rate), int((up + car_h/2)*column_rate)]]
            # ego_info = [
            # [int((left + gt_y[frame] - car_w/2)*row_rate), int((up - gt_x[frame] - car_h/2)*column_rate)], 
            # [int((left + gt_y[frame] + car_w/2)*row_rate), int((up - gt_x[frame] - car_h/2)*column_rate)], 
            # [int((left + gt_y[frame] + car_w/2)*row_rate), int((up - gt_x[frame] + car_h/2)*column_rate)], 
            # [int((left + gt_y[frame] - car_w/2)*row_rate), int((up - gt_x[frame] + car_h/2)*column_rate)]]
            rectangle = np.array(ego_info) 
            
            if (rectangle[0][0]+rectangle[1][0])/2<image_w and (rectangle[0][1]+rectangle[1][1])/2<image_h:
                cv2.fillConvexPoly(img, rectangle, (0,0,0))    # 画主车        
            #    cv2.imwrite(f'{bev_path}/{str(car_id)+"_"+str(ts)}.jpg', img)
            return img
        except:
            print('first_Image: {} has wrong!'.format(ts))
    # GIF
    def prepare_img_list(self,img_path):
        file_name = os.listdir(img_path)
        img_list = sorted(file_name)
        # img_list = sorted(file_name,key = lambda i:int("".join(list(filter(str.isdigit, i)))))
        gif_images = []
        for img_name in img_list:
            if img_name.find('.jpg') != -1:
                file_name = os.path.join(img_path,img_name)
                gif_images.append(file_name)
        return gif_images

    def create_gif(self,image_list, gif_name, duration = 0.05):
        '''
        参数
        :1. image_list: 这个列表用于存放生成动图的图片
        :2. gif_name: 字符串，所生成gif文件名，带.gif后缀
        :3. duration: 图像间隔时间
        '''
        frames = []
        for image_name in tqdm(image_list,desc='create gif'):
            frames.append(imageio.imread(image_name))
        imageio.mimsave(gif_name, frames, 'GIF', fps=10)

        return

    def plot_src(self,n):
        # 画真值和预测
        # src_path = '/data3/data_haomo/m1/csv/1212/eval_scene/pretrain_turn_right_seq2seq_11.csv'
        src_path = '/data/wangyue/model_one/tmp/val_set.csv'
        traj_path = '/data/wangyue/model_one/experiment/mmrt_v1.1/visz/test0103.csv'
        bev_path='/data/wangyue/model_one/experiment/ptrt_v1.6/visz/imgs/'
        ade_info_df = pd.read_csv('/data/wangyue/model_one/experiment/mmrt_v1.1/visz/test_ade.csv')
        ade = ade_info_df['ADE'].values

        src_df = pd.read_csv(src_path)[n:n+130]
        traj_df = pd.read_csv(traj_path)

        dist_x =  src_df['waypoint_x'].values.tolist()[80:]
        gt_x = self.sum_col(dist_x)
        dist_y = src_df['waypoint_y'].values.tolist()[80:]
        gt_y = self.sum_col(dist_y)
        
        pred_x =  traj_df['pred_x'].values[:50].tolist()
        pred_x = self.sum_col(pred_x)
        pred_y =  traj_df['pred_y'].values[:50].tolist()
        pred_y = self.sum_col(pred_y)


        car_id = src_df['car_id'].values.tolist()[80:]
        ts = src_df['ts'].values.tolist()[80:]

        batch_id = n
        if not os.path.exists(os.path.join(bev_path, f'batch_{batch_id}')):
            os.mkdir(os.path.join(bev_path, f'batch_{batch_id}'))
        for i in range(len(gt_x)):
            curr_ts, id = ts[i], car_id[i]

            obs_info = src_df[src_df['ts'].isin([curr_ts])]['obs_info']
            lane_info =  src_df[src_df['ts'].isin([curr_ts])]['lane_info']
            
            #自车动
            # img = self.bev_drawing2(id, curr_ts, i, angle, obs_info, lane_info, bev_path, up = 150-dist_x[i], down = 20+dist_x[i], left = 25-dist_y[i], right = 25+dist_y[i], row_rate = 5, column_rate = 5, color = 1, obs = True, line = True, ego_w = 2, ego_h = 5, is_rewritted = False)
            #自车不动
            img = self.bev_drawing2(id, curr_ts, obs_info, lane_info, bev_path, 
            up = 100, down = 20, left = 25, right = 25, row_rate = 5, column_rate = 5, 
            color = 1, obs = True, line = True, ego_w = 2, ego_h = 5, is_rewritted = False)
            
            img = cv2.putText(img,'ade: '+str(ade[0])[:8],(10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv2.LINE_AA)
            img = cv2.putText(img,'id: '+str(n),(10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv2.LINE_AA)
            
            self.bev_drawing_trac(batch_id, i ,pred_x, pred_y, bev_path, img, gt_x, gt_y,
             up = 100 + dist_x[i], down = 20 - dist_x[i], left = 25 + dist_y[i], right = 25 + dist_y[i], 
             row_rate = 5, column_rate = 5, ego_w = 2, ego_h = 5)
            # self.bev_drawing_trac(batch_id, i ,pred_x, pred_y, bev_path, img, gt_x, gt_y, up = 100, down = 20, left = 25, right = 25, row_rate = 5, column_rate = 5, ego_w = 2, ego_h = 5)
        
        img_path = os.path.join(bev_path, f'batch_{batch_id}')
        image_list = self.prepare_img_list(img_path)
        self.create_gif(image_list, os.path.join(img_path, "traj.gif"), duration=1)
        print(f"bathc_id:{batch_id},done!")  
    
    def plot(self, cfg, src_path, output_path, ade_path):
        # 批量画gif 
        src_df = pd.read_csv(src_path)
        output_df = pd.read_csv(output_path)
        ade_df = pd.read_csv(ade_path)
        
        file_name = os.path.splitext(os.path.basename(output_path))[0]
        bev_path = os.path.join(cfg.DIR.VISUAL,cfg.EXP.MODEL_NAME,'visz','imgs',file_name)
        if not os.path.exists(bev_path):
            os.makedirs(bev_path,exist_ok=True)
        cnt = 0
        for i in tqdm(range(0,output_df.shape[0],cfg.DATA.FRAME.TGT_FRAME_LEN*10),desc='plotting'):
            # 获取预测轨迹坐标
            pred_df = output_df[i:i + cfg.DATA.FRAME.TGT_FRAME_LEN]
            pred_x =  pred_df['pred_x'].values.tolist()
            pred_x = self.sum_col(pred_x)
            pred_y =  pred_df['pred_y'].values.tolist()
            pred_y = self.sum_col(pred_y)
            
            dist_x =  pred_df['gt_x'].values.tolist()
            gt_x = self.sum_col(dist_x)
            dist_y = pred_df['gt_y'].values.tolist()
            gt_y = self.sum_col(dist_y)

            car_id = pred_df['car'].values.tolist()
            ts = pred_df['ts'].values.tolist()
            ade = ade_df['ADE'][i//(50*cfg.EVAL.BATCH_SIZE)]

            if not os.path.exists(os.path.join(bev_path, f'scene_{cnt}')):
                os.mkdir(os.path.join(bev_path, f'scene_{cnt}'))

            for j in range(len(pred_x)):
                curr_ts, id = ts[j], car_id[j]
                obs_info = src_df[src_df['ts'].isin([curr_ts])]['obs_info']
                lane_info =  src_df[src_df['ts'].isin([curr_ts])]['lane_info']

                img = self.bev_drawing2(id, curr_ts, obs_info, lane_info, bev_path, 
                up = 100, down = 20, left = 25, right = 25, row_rate = 5, column_rate = 5, 
                color = 1, obs = True, line = True, ego_w = 2, ego_h = 5, is_rewritted = False)
                
                img = cv2.putText(img,'ade: '+str(ade)[:8],(10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv2.LINE_AA)
                
                self.bev_drawing_trac(cnt, j ,pred_x, pred_y, bev_path, img, gt_x, gt_y,
                up = 100 + dist_x[j], down = 20 - dist_x[j], left = 25 + dist_y[j], right = 25 + dist_y[j], 
                row_rate = 5, column_rate = 5, ego_w = 2, ego_h = 5)

            img_path = os.path.join(bev_path, f'scene_{cnt}')
            image_list = self.prepare_img_list(img_path)
            self.create_gif(image_list, os.path.join(img_path, "traj.gif"), duration=1)
            print(f"scene_id:{cnt},done!")  
            cnt += 1
            
            if cnt == 20:
                break 

    def eval_plot(self,cfg,src_df,output_path):
        # 对接eval_service/eval.py
        output_df = pd.read_csv(output_path)

        file_name = os.path.splitext(os.path.basename(output_path))[0]
        bev_path = os.path.join(cfg.eval_output_dir,'imgs')
        if not os.path.exists(bev_path):
            os.makedirs(bev_path,exist_ok=True)
        cnt = 0
        for i in range(0,output_df.shape[0],cfg.DATA.FRAME.TGT_FRAME_LEN*10):
            # 获取预测轨迹坐标
            pred_df = output_df[i:i + cfg.DATA.FRAME.TGT_FRAME_LEN]
            pred_x =  pred_df['pred_x'].values.tolist()
            # pred_x = self.sum_col(pred_x)
            pred_y =  pred_df['pred_y'].values.tolist()
            # pred_y = self.sum_col(pred_y)
            
            gt_x =  pred_df['gt_x'].values.tolist()
            # gt_x = self.sum_col(dist_x)
            gt_y = pred_df['gt_y'].values.tolist()
            # gt_y = self.sum_col(dist_y)

            car_id = pred_df['car'].values.tolist()
            ts = pred_df['ts'].values.tolist()
            ade = pred_df['ADE'][i]

            if not os.path.exists(os.path.join(bev_path, f'scene_{cnt}')):
                os.mkdir(os.path.join(bev_path, f'scene_{cnt}'))
            
            for j in range(len(pred_x)):
                curr_ts, id = ts[j], car_id[j]
                obs_info = src_df[src_df['ts'].isin([curr_ts])]['obs_info']
                lane_info =  src_df[src_df['ts'].isin([curr_ts])]['lane_info']

                img = self.bev_drawing2(id, curr_ts, obs_info, lane_info, bev_path, 
                up = 100, down = 20, left = 25, right = 25, row_rate = 5, column_rate = 5, 
                color = 1, obs = True, line = True, ego_w = 2, ego_h = 5, is_rewritted = False)
                
                img = cv2.putText(img,'ade: '+str(ade)[:8],(10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv2.LINE_AA)
                
                self.bev_drawing_trac(cnt, j ,pred_x, pred_y, bev_path, img, gt_x, gt_y,
                up = 100, down = 20, left = 25, right = 25, 
                row_rate = 5, column_rate = 5, ego_w = 2, ego_h = 5)

            img_path = os.path.join(bev_path, f'scene_{cnt}')
            image_list = self.prepare_img_list(img_path)
            self.create_gif(image_list, os.path.join(img_path, "traj.gif"), duration=1)
            print(f"scene_id:{cnt},done!")  
            cnt += 1

    def cont_plot(self,cfg,src_path,output_path):
        # 画连续场景
        src_df = pd.read_csv(src_path)
        output_df = pd.read_csv(output_path)
        file_name = os.path.splitext(os.path.basename(src_path))[0]
        bev_path = os.path.join(cfg.DIR.EVAL_DIR,cfg.EXP.MODEL_NAME,file_name,'imgs_1')
        if not os.path.exists(bev_path):
            os.makedirs(bev_path,exist_ok=True)

        cnt = 1  # 文件夹名 scene_{cnt}
        
        start = 40000
        end = 60000
        length = end - start
        for i in tqdm(range(start,end, cfg.DATA.FRAME.TGT_FRAME_LEN), total=length//50, desc='create bev'):   # 控制帧数
            # 获取预测轨迹坐标
            pred_df = output_df[i:i + cfg.DATA.FRAME.TGT_FRAME_LEN]
            pred_x =  pred_df['pred_x'].values.tolist()
            # pred_x = self.sum_col(pred_x)
            pred_y =  pred_df['pred_y'].values.tolist()
            # pred_y = self.sum_col(pred_y)
            
            gt_x =  pred_df['gt_x'].values.tolist()
            # gt_x = self.sum_col(dist_x)
            gt_y = pred_df['gt_y'].values.tolist()
            # gt_y = self.sum_col(dist_y)

            car_id = pred_df['car'].values.tolist()
            ts = pred_df['ts'].values.tolist()
            # ade = ade_df['ADE'][i//(50*cfg.EVAL.BATCH_SIZE)]
            ade = pred_df['ADE'].values.tolist()

            if not os.path.exists(os.path.join(bev_path, f'scene_{cnt}')):
                os.mkdir(os.path.join(bev_path, f'scene_{cnt}'))
            

            curr_ts, id, ade = ts[0], car_id[0], ade[0]
            obs_info = src_df[src_df['ts'].isin([curr_ts])]['obs_info']
            lane_info =  src_df[src_df['ts'].isin([curr_ts])]['lane_info']

            img = self.bev_drawing2(id, curr_ts, obs_info, lane_info, bev_path, 
            up = 100, down = 20, left = 25, right = 25, row_rate = 5, column_rate = 5, 
            color = 1, obs = True, line = True, ego_w = 2, ego_h = 5, is_rewritted = False)
            
            img = cv2.putText(img,'ade: '+str(ade)[:8],(10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv2.LINE_AA)
            img = cv2.putText(img,'ts: ' + str(curr_ts),(10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv2.LINE_AA)

            self.bev_drawing_trac(cnt, i//50 ,pred_x, pred_y, bev_path, img, gt_x, gt_y,
            up = 100 , down = 20 , left = 25 , right = 25 , 
            row_rate = 5, column_rate = 5, ego_w = 2, ego_h = 5)

        img_path = os.path.join(bev_path, f'scene_{cnt}')
        image_list = self.prepare_img_list(img_path)
        self.create_gif(image_list, os.path.join(img_path, f"traj_{i}.gif"), duration=1)
        # remove pred_*.jpg
        os.system(f'rm -f {img_path}/pred_*.jpg')

        print(f"scene_id:{cnt},done!")  

    def multi_bev_plot(self,data_path,save_path):
        # 生成BEV图
        df = pd.read_csv(data_path)
        for i in tqdm(range(len(df))):
            car_id = df.loc[i]['car_id']
            curr_ts = df.loc[i]['ts']
            obs_info = df[df['ts'].isin([curr_ts])]['obs_info']
            lane_info =  df[df['ts'].isin([curr_ts])]['lane_info']
            img = self.bev_drawing2(car_id, curr_ts, obs_info, lane_info,
             save_path, up = 50, down = 20, left = 25, right = 25, row_rate = 5, 
             column_rate = 5, color = 1, obs = True, line = True, ego_w = 2, 
             ego_h = 5, is_rewritted = False)
            cv2.imwrite(os.path.join(save_path,f'{car_id}_{str(curr_ts)}.jpg'), img)




if __name__ == "__main__":
    cfg = load_yaml2cfg("/data/wangyue/model_one/config_service/m1_config_base.yaml")
    visz_block = Visualization(cfg)

    # data_path = cfg.DATA_SOURCE.TXT_PATH
    # save_path = cfg.DATA_SOURCE.IMG_DIR
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path,exist_ok=True)

    # visz_block.multi_bev_plot(data_path,save_path)s

    src_path = cfg.EVAL.DATA_PATH
    output_path = '/data/wangyue/model_one/experiment/eval/ptrt_v1.1/turn_left/output_traj.csv'
    # visz_block.plot(cfg,src_path,output_path,ade_path)
    visz_block.cont_plot(cfg,src_path,output_path)

        


  
