'''
Author: wangyue
Date: 2022-08-24 17:53:17
LastEditors: wangyue
LastEditTime: 2022-11-08 11:03:46
FilePath: /model_one/model_service/train.py
Description:

Copyright (c) 2022 by Haomo, All Rights Reserved.
'''
import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch
from tqdm import tqdm
import time
import copy
import argparse
from utils.cfg_utils import *
from model_base.multi_model_encoder import *
from model_base.model_utils import *
from model_base.traj_evaluator import TrajEvaluator

class Trainer:

    def __init__(self, model, cfg, loader_dict, nums_dict):
        self.cfg = cfg
        self.model = model
        self.loader_dict = loader_dict
        self.nums_dict = nums_dict
        # get the name of net
        self.model_name = cfg.EXP.MODEL_NAME
        # Initialize necessary services
        cfg.log_dir = os.path.join(cfg.DIR.LOG, cfg.EXP.MODEL_NAME)
        cfg.visual_dir = os.path.join(
            cfg.DIR.VISUAL, cfg.EXP.MODEL_NAME, 'train')
        cfg.model_dir = os.path.join(cfg.DIR.MODEL_DIR, cfg.EXP.MODEL_NAME)

        self.model_utils = ModelUtils(cfg)
        # self.criterion = get_loss_function(cfg, samples_per_class)
        self.criterion = nn.HuberLoss() 
        self.optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.TRAIN.OPTIMIZER.LR)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, cfg.TRAIN.OPTIMIZER.STEP_SIZE, gamma = cfg.TRAIN.OPTIMIZER.GAMMA)
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.model_utils.set_random_seed(attach=self.rank)
        self.best_epoch_based_mse_loss = float('inf')
        self.best_epoch = 1
        self.evalutor = TrajEvaluator()
        self.start_epoch = 1

        if self.rank == 0:
            # Initialize other services
            self.model_utils.params_count(self.model)
            self.evaluator = EvaluateUtils(cfg)
            self.visual_utils = VisualizationUtils(cfg)
            self.mylogger = LoggingUtils(cfg)

    def ddp_train(self):
        start_time = time.time()
        if self.rank == 0:
            print('Start time: ', datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"))
            self.mylogger.info_logger('🛫🛫🛫🛫🛫🛫 Training Start 🛫🛫🛫🛫🛫🛫')

        # model recover from checkpoint

        if self.cfg.TRAIN.RECOVER.IS_RECOVER:
            ckpt_path = ''
            if self.cfg.TRAIN.RECOVER.IS_AUTO_RECOVER and os.path.exists(self.cfg.model_dir): # 自动化找最新版本checkpoint
                model_files = os.listdir(self.cfg.model_dir)
                ckpt_files = [model_file for model_file in model_files if model_file.find('model_ckpt_epoch')!=-1]
                if ckpt_files and len(ckpt_files) > 0:
                    ckpt_path = os.path.join(self.cfg.model_dir,sorted(ckpt_files,reverse=True)[0])# 获取最大epoch对应checkpoint文件
            else:# 手动指定某个版本checkpoint
                ckpt_path = os.path.join(self.cfg.model_dir,f'model_ckpt_epoch_{self.cfg.TRAIN.RECOVER.RECOVER_STEP}.pth')
    
            self.do_recover_model(ckpt_path)

        for epoch in range(self.start_epoch, self.cfg.TRAIN.EPOCHS + 1):
            torch.cuda.empty_cache()
            self.loader_dict['train'].sampler.set_epoch(epoch)
            self.model.train() # train mode
            total_train_loss = 0
            epoch_train_loss = 0
            for iter_idx, batch_data in enumerate(tqdm(self.loader_dict['train'], desc='Distributed {} | Epoch {} is training'.format(self.model_name, epoch))):
                self.optimizer.zero_grad() # 当有模型参数分组，用这种方法
                src_input,tgt_gt = batch_data[0].to(self.model.device), batch_data[1].to(self.model.device)
                src_input = src_input.permute([1, 0, 2]).contiguous()  # [seq,batch,feature_size]
                tgt_gt = tgt_gt.permute([1, 0, 2]).contiguous()  # [seq,batch,feature_size]
                tgt_pred = self.model(src = src_input)
                tgt_pred = tgt_pred[-self.cfg.DATA.FRAME.TGT_FRAME_LEN:]
                tgt_gt = tgt_gt[-self.cfg.DATA.FRAME.TGT_FRAME_LEN:]
                train_loss = self.criterion(tgt_pred,tgt_gt) # for masked LM ;masked_tokens [6,5]
                # dist.reduce(train_loss, dst=0, op=dist.ReduceOp.SUM)
                if self.rank == 0 and torch.cuda.device_count() > 1:
                    train_loss = train_loss.mean() # mean() to average on multi-gpu.
                train_loss_val = train_loss.cpu().item() # 从cpu获取具体值
                
                epoch_train_loss += train_loss_val/self.cfg.TRAIN.BATCH_SIZE
                total_train_loss += train_loss_val

                # save model
                if(iter_idx > 0 and iter_idx % 199 == 0):
                    if self.rank == 0:
                        self.mylogger.info_logger(
                            f'Epoch:{epoch},epoch train loss={epoch_train_loss},lr={self.scheduler.get_last_lr()[0]}')
                        self.mylogger.info_logger(
                            f'Epoch:{epoch},best train loss={self.best_epoch_based_mse_loss},lr={self.scheduler.get_last_lr()[0]}')
                        avg_train_loss = total_train_loss/len(self.loader_dict['train'])
                        self.mylogger.info_logger(
                            f'Epoch:{epoch},avg train loss={avg_train_loss},lr={self.scheduler.get_last_lr()[0]}')
                        
                if train_loss_val <= self.best_epoch_based_mse_loss:
                    self.best_epoch_based_mse_loss = train_loss_val
                    self.best_epoch = epoch
                    if self.rank == 0:
                        self.mylogger.critical_logger(
                            f'Epoch:{epoch},best loss={self.best_epoch_based_mse_loss},lr={self.scheduler.get_last_lr()[0]}')
                        self.model_utils.model_to_save(
                            self.model, self.optimizer,self.scheduler,epoch, self.best_epoch_based_mse_loss)
                        
                    
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.scheduler.step()

            # 可视化日志记录
            if self.rank == 0:
                train_metrics_dict = {
                    'best_train_loss': self.best_epoch_based_mse_loss,
                    'epoch_train_loss': epoch_train_loss,
                    'avg_train_loss': total_train_loss/len(self.loader_dict['train'])
                }
                self.visual_utils.record_metrics(
                    epoch, train_metrics_dict, None)

            
            # eval验证模块
            if self.cfg.EVAL.IS_EVAL:
                if epoch%self.cfg.EVAL.EVAL_EPOCH == 0:
                    self.ddp_eval(self.model,epoch)

            if self.rank == 0:
                self.mylogger.info_logger(
                    '💺💺💺 Epoch {} Done 💺💺💺'.format(epoch))
                # 每轮遍历完毕，将模型保存起来，因为上面保存模型仅保存loss最小，而在训练过程中会出现当前epoch loss没有降低
                if self.best_epoch != epoch:
                    self.model_utils.model_to_save(
                                self.model, self.optimizer,self.scheduler,epoch, self.best_epoch_based_mse_loss)

        torch.cuda.synchronize()
        if self.rank == 0:
            self.mylogger.info_logger('🛬🛬🛬🛬🛬🛬 Training End 🛬🛬🛬🛬🛬🛬')
            print('End time: ', datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"))
            print('Model train cost {:.4f}s.'.format(
                time.time() - start_time))
    
    def do_recover_model(self,ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f'模型文件{ckpt_path}不存在，无法正常加载，将从头开始训练！')
            return
        ckpt_data = torch.load(ckpt_path)
        ckpt_epoch = ckpt_data['epoch']
        self.start_epoch = ckpt_epoch + 1
        self.best_epoch = self.start_epoch
        self.best_epoch_based_mse_loss = ckpt_data['best_loss']
        self.model.load_state_dict(ckpt_data['state_dict'])
        self.optimizer.load_state_dict(ckpt_data['optimizer'])
        self.scheduler.load_state_dict(ckpt_data['lr_scheduler'])
        print(f'模型加载 epoch = {ckpt_epoch} 成功！')


    def ddp_eval(self,model,epoch):
        model = copy.deepcopy(model)
        model.eval()  # turn on evaluation mode
        device= self.model.device
        if self.rank == 0:
            # reset records for valid dataset
            output_all, label_all = None, None
            output_list = [torch.zeros(self.cfg.DATA.FRAME.TGT_FRAME_LEN,self.cfg.EVAL.BATCH_SIZE,self.cfg.MODEL.PRETRAIN.OUT_FEATURES_SIZE).cuda() for _ in range(self.world_size)]
            label_list = [torch.zeros(self.cfg.DATA.FRAME.TGT_FRAME_LEN,self.cfg.EVAL.BATCH_SIZE,self.cfg.MODEL.PRETRAIN.OUT_FEATURES_SIZE).cuda() for _ in range(self.world_size)]
        else:
            output_list, label_list = None, None
        
        with torch.no_grad():
            ade,fde = 0.0,0.0
            cnt = 0
            for iter_idx, batch_data in enumerate(tqdm(self.loader_dict['valid'], desc='Distributed {} | Epoch {} is evaluating'.format(self.model_name, epoch))):
                if None == batch_data[0]: continue
                src_input,tgt_gt = batch_data[0].to(device).permute([1,0,2]).contiguous(), \
                                  batch_data[1].to(device).permute([1,0,2]).contiguous()
                
                tgt_pred = model(src = src_input)
                tgt_pred = tgt_pred[-self.cfg.DATA.FRAME.TGT_FRAME_LEN:]
                tgt_gt = tgt_gt[-self.cfg.DATA.FRAME.TGT_FRAME_LEN:]
                
                dist.gather(tensor=tgt_pred, gather_list=output_list, dst=0)
                dist.gather(tensor=tgt_gt, gather_list=label_list, dst=0)
                
                if self.rank == 0:
                    # record predict results
                    if output_all is None:
                        output_all = torch.cat(output_list, dim=0)
                        label_all = torch.cat(label_list, dim=0)
                    else:
                        output_all = torch.cat([output_all, torch.cat(output_list, dim=0)], dim=0)
                        label_all = torch.cat([label_all, torch.cat(label_list, dim=0)], dim=0)

                    metric = self.evalutor.compute_prediction_metrics(output_all[:,:,:2].unsqueeze(1), label_all[:,:,:2])
                    ade += metric["ADE"]  #整批数据ade
                    fde += metric["FDE"]
                    print(f"batch_ids:{iter_idx},metric:{metric}")
                    cnt += 1
            if self.rank == 0:
                print(f"valid of epoch:{epoch},info:  ADE:{ade/cnt},FDE:{fde/cnt}")
                self.mylogger.info_logger(f"valid of epoch:{epoch},info:  ADE:{ade/cnt},FDE:{fde/cnt}")
                valid_metrics_dict = {
                        'ade': ade/cnt,
                        'fde': fde/cnt,
                    }
                self.visual_utils.record_metrics(epoch, None,valid_metrics_dict)
            
            if self.rank == 0:
                # reset records for valid dataset
                output_all, label_all = None, None
                output_list = [torch.zeros(self.cfg.DATA.FRAME.TGT_FRAME_LEN,self.cfg.EVAL.BATCH_SIZE,self.cfg.MODEL.PRETRAIN.OUT_FEATURES_SIZE).cuda() for _ in range(self.world_size)]
                label_list = [torch.zeros(self.cfg.DATA.FRAME.TGT_FRAME_LEN,self.cfg.EVAL.BATCH_SIZE,self.cfg.MODEL.PRETRAIN.OUT_FEATURES_SIZE).cuda() for _ in range(self.world_size)]
            else:
                output_list, label_list = None, None
