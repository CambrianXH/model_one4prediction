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
from model_base.model_server import get_trm_model_plus
from data_service.seq2seq_dataloader import get_data
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
        cfg.log_dir = os.path.join(cfg.DIR.ROOT, cfg.EXP.MODEL_NAME, 'logs')
        cfg.visual_dir = os.path.join(
            cfg.DIR.VISUAL, cfg.EXP.MODEL_NAME, 'visz/train')
        cfg.model_dir = os.path.join(cfg.DIR.MODEL_DIR, cfg.EXP.MODEL_NAME)

        self.model_utils = ModelUtils(cfg)
        # self.criterion = get_loss_function(cfg, samples_per_class)
        self.criterion = nn.MSELoss() 
        self.optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.TRAIN.OPTIMIZER.LR)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, cfg.TRAIN.OPTIMIZER.STEP_SIZE, gamma = cfg.TRAIN.OPTIMIZER.GAMMA)
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.model_utils.set_random_seed(attach=self.rank)
        self.best_epoch_based_mse_loss = float('inf')
        self.best_epoch = 1
        self.evalutor = TrajEvaluator()
        self.is_img = cfg.DATA.IS_IMG
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



        for epoch in range(1, self.cfg.TRAIN.EPOCHS + 1):
            torch.cuda.empty_cache()
            # train mode
            self.loader_dict['train'].sampler.set_epoch(epoch)
            self.model.train()
            total_train_loss = 0
            epoch_train_loss = 0
            for iter_idx, batch_data in enumerate(tqdm(self.loader_dict['train'], desc='Distributed {} | Epoch {} is training'.format(self.model_name, epoch))):
                # print(f'iter_idx={iter_idx}')
                self.optimizer.zero_grad() # 当有模型参数分组，用这种方法
                src_input,tgt_gt = batch_data[0].to(self.model.device), batch_data[1].to(self.model.device)
                start_symbol = torch.ones(tgt_gt.size(0),1,tgt_gt.size(2)) * -1000
                start_symbol = start_symbol.to(self.model.device)
                tgt_in = torch.cat([start_symbol,tgt_gt],dim=1)[:,:-1,:]
                input = (src_input,tgt_in)
                if self.is_img:
                    seq_img_feature = batch_data[3].to(self.model.device)
                else:
                    seq_img_feature = None
                tgt_pred = self.model(input,seq_img_feature=seq_img_feature)

                train_loss = self.criterion(tgt_pred,tgt_gt) # for masked LM ;masked_tokens [6,5]
                dist.reduce(train_loss, dst=0, op=dist.ReduceOp.SUM)
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

    def ddp_eval(self, model,epoch):
        model = copy.deepcopy(model)
        model.eval()  # turn on evaluation mode
        device= self.model.device
        with torch.no_grad():
            ade,fde = 0.0,0.0
            cnt = 0
            for iter_idx, batch_data in enumerate(tqdm(self.loader_dict['valid'], desc='Distributed {} | Epoch {} is evaluating'.format(self.model_name, epoch))):
                src_input,tgt_gt = batch_data[0].to(device), \
                                   batch_data[1].to(device)
                if self.is_img:
                    seq_img_feature = batch_data[3].to(self.model.device)
                else:
                    seq_img_feature = None
                tgt_pred = self.get_forward_step_decode(self.cfg,model,src_input,seq_img_feature,[-1000,-1000])

                if self.rank == 0:
                    tgt_pred = tgt_pred[:,:,:2] # x/y坐标
                    tgt_gt = tgt_gt[:,:,:2] # x/y坐标
                    metric = self.evalutor.compute_prediction_metrics(tgt_pred.unsqueeze(1), tgt_gt)
                    ade += metric["ADE"]  #整批数据ade
                    fde += metric["FDE"]
                    cnt += 1
            if self.rank == 0:
                print(f"valid of epoch:{epoch},info:  ADE:{ade/cnt},FDE:{fde/cnt}")
                self.mylogger.info_logger(f"valid of epoch:{epoch},info:  ADE:{ade/cnt},FDE:{fde/cnt}")
                valid_metrics_dict = {
                        'ade': ade/cnt,
                        'fde': fde/cnt,
                    }
                self.visual_utils.record_metrics(epoch, None,valid_metrics_dict)


    def get_forward_step_decode(self, cfg, model, enc_input, seq_img_feature, start_symbol):
        enc_outputs = model.module.encode_src(enc_input,seq_img_feature)
        next_symbol = torch.tensor(start_symbol).view(1,1,2).repeat(cfg.EVAL.BATCH_SIZE,1,1)
        dec_input = next_symbol.type_as(enc_input.data)
        for i in range(0,cfg.DATA.FRAME.TGT_FRAME_LEN):
            dec_outputs = model.module.decode_tgt(tgt=dec_input,memory=enc_outputs)
            next_symbol = dec_outputs[:,-1,:].unsqueeze(1)
            dec_input = torch.cat([dec_input,next_symbol],dim=1)
        output = model((enc_input,dec_input[:,:-1,:]),seq_img_feature)
        return output
