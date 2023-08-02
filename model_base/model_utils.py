'''
Author: wangyue
Date: 2022-07-05 10:19:00
LastEditors: aiwenjie aiwenjie20@outlook.com
LastEditTime: 2022-11-08 10:17:52
FilePath: /scene_recognition/model_service/utils.py
Description: model utils for train and valid.

Copyright (c) 2022 by Haomo, All Rights Reserved.
'''
import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import torch
import random
import shutil
import logging
import datetime
import numpy as np
from torchinfo import summary
# mpl.use('Agg')
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import _Loss
import copy

def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors



class ModelUtils:
    '''
    - check out work directory.
    - fix random seed.
    - save model and other parameters.
    - load model.
    - compute all parameters and trainable parameters.
    '''

    def __init__(self, cfg, distributed=True):
        self.cfg = cfg
        self.log_dir = cfg.log_dir
        self.model_dir = cfg.model_dir
        self.seed = cfg.EXP.SEED
        
        if distributed and dist.get_rank() == 0:
            self.check_dir()

    def check_dir(self):
        '''check out log directory, model directory, checkpoint directory exist or not.'''
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir,exist_ok=True)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir,exist_ok=True)
    
    def set_random_seed(self, attach):
        '''set random seed for all environment.'''
        random.seed(self.seed + attach)
        np.random.seed(self.seed + attach)
        torch.manual_seed(self.seed + attach)
        torch.cuda.manual_seed(self.seed + attach)
        torch.cuda.manual_seed_all(self.seed + attach)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def save_checkpoint(self, cur_model, optimizer, indicator):
        '''save current model.'''
        checkpoint_info = {
            'model_state_dict': cur_model.module.state_dict(),
            'random_seed': self.cfg.EXP.SEED,
            # 'finetune': self.cfg.MODEL.TEXT.FINETUNE,
            'batch_size': self.cfg.TRAIN.BATCH_SIZE,
            'loss_name': self.cfg.TRAIN.LOSS.NAME,
            'optimizer_name': self.cfg.TRAIN.OPTIMIZER.NAME,
            'optimizer_state_dict': optimizer.state_dict(),
        }
        checkpoint_name = '{}_{}.pth'.format(self.cfg.EXP.MODEL_NAME, indicator)
        torch.save(checkpoint_info, os.path.join(self.ckpt_dir, checkpoint_name))

    def load_checkpoint(self, model, indicator='accuracy'):
        '''load model in specific model.'''
        checkpoint_name = '{}_{}_{}.pth'.format(self.cfg.MODEL.NAME, self.cfg.VERSION, indicator)
        checkpoint_info = torch.load(os.path.join(self.cfg.DIR.CHECKPOINT, checkpoint_name),map_location='cpu')
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint_info['model_state_dict'].items() if k in model.state_dict()})
        # model.load_state_dict(checkpoint_info['model_state_dict'])

    def params_count(self, model):
        '''compute all parameters and trainable parameters.'''
        num_trainable_params = np.sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_all_params = np.sum(p.numel() for p in model.parameters())
        print('*' * 80)
        print('Number of Parameters')
        print('*' * 80)
        print('{:^16}\t{:^16}\t{:^16}' .format('Total Parameters', 'Trainable Parameters', 'Untrainable Parameters'))
        print('{:^16d}\t{:^16d}\t{:^16d}' .format(num_all_params, num_trainable_params, num_all_params -num_trainable_params))
        print('*' * 80)

   
    def model_to_save(self,model,optimizer,lr_scheduler,epoch,best_loss,format='bin'):
        ''' save a trained model'''
        # model_to_save = model.module if hasattr(
        #     model, 'module') else model  # Only save the model it-self
        model_to_save = copy.deepcopy(model)
        # save model state dict
        model_file_state_dict = os.path.join(self.model_dir, 
        f'model_state_dict_epoch_{epoch}.{format}')
        torch.save(model_to_save.state_dict(), model_file_state_dict)

        # save model info to recovery model
        state = {
        'epoch': epoch,
        'state_dict': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'best_loss': best_loss,
        }
        model_file_ckpt = os.path.join(self.model_dir, f'model_ckpt_epoch_{epoch}.pth')
        torch.save(state, model_file_ckpt)

    def get_learning_rate(self,optimizer):
        lr=[]
        for param_group in optimizer.param_groups:
            lr +=[ param_group['lr'] ]
        return lr

class EvaluateUtils:
    '''
    - compute metrics for train and valid.
    '''
    def __init__(self, cfg):
        self.cfg = cfg

    def compute_metrics(self, output_all, label_all):
        '''compute metircs for all categories'''

        # compute accuracy, precision, recall, f1_score
        # acc = accuracy(preds=output_all, target=label_all, num_classes=self.cfg.DATA.NUM_CLASSES)
        # pre = precision(preds=output_all, target=label_all, average=None, num_classes=self.cfg.DATA.NUM_CLASSES)
        # rec = recall(preds=output_all, target=label_all, average=None, num_classes=self.cfg.DATA.NUM_CLASSES)
        # f1 = f1_score(preds=output_all, target=label_all, average=None, num_classes=self.cfg.DATA.NUM_CLASSES)
        # index_dict = {
        #     'accuracy': acc.cpu().numpy(),
        #     'precision': pre.cpu().numpy(),
        #     'recall': rec.cpu().numpy(),
        #     'f1_score': f1.cpu().numpy() 
        # }

        # # compute confusion_matrix
        # cmtx = confusion_matrix(preds=output_all, target=label_all, num_classes=self.cfg.DATA.NUM_CLASSES).cpu().numpy()

        # return index_dict, cmtx
        
class VisualizationUtils:
    '''
    - show loss and other metrics for train and valid.
    - show model structure and number of parameters.
    '''

    def __init__(self, cfg):
        self.cfg = cfg
        self.visual_dir = cfg.visual_dir
        # check是否要创建目录
        if not os.path.exists(self.visual_dir):
            os.makedirs(self.visual_dir,exist_ok=True)
        
        self.writer = SummaryWriter(self.visual_dir)
        if dist.get_rank == 0:
            self.visual_dir_reset()

    
    def show_model(self, model):
        '''show model structure'''
        summary(model)

    def record_metrics(self, epoch, train_metrics_dict, valid_metrics_dict):
        '''show metrics, such as  train loss.'''
        if train_metrics_dict and train_metrics_dict.get('epoch_train_loss',None) != None:
            self.writer.add_scalar('epoch train loss', 
                train_metrics_dict['epoch_train_loss'],
                epoch
            )
        if train_metrics_dict and train_metrics_dict.get('best_train_loss',None) != None:
            self.writer.add_scalar('best train loss', 
                train_metrics_dict['best_train_loss'],
                epoch
            )
        if train_metrics_dict and train_metrics_dict.get('avg_train_loss',None) != None:
            self.writer.add_scalar('avg train loss', 
                train_metrics_dict['avg_train_loss'],
                epoch
            )
        if train_metrics_dict:
            if epoch == self.cfg.TRAIN.EPOCHS:
                self.writer.close()
        
        if valid_metrics_dict and valid_metrics_dict.get('ade',None) != None:
            self.writer.add_scalar('each batch of valid ade', 
                valid_metrics_dict['ade'],
                epoch
            )
        if valid_metrics_dict and valid_metrics_dict.get('fde',None) != None:
            self.writer.add_scalar('each batch of valid fde', 
                valid_metrics_dict['fde'],
                epoch
            )

    def manual_close_tb_writer(self):
        self.writer.close()

    def visual_dir_reset(self):
        '''clear the visual directory'''
        file_list = os.listdir(self.visual_dir)
        for file in file_list:
            shutil.rmtree(os.path.join(self.visual_dir, file))


def get_loss_function(cfg, samples_per_class, reduction='mean'):
    if cfg.TRAIN.LOSS.NAME == 'cbf_softmax_cross_entropy':
        return ClassBalancedFocalSoftmaxCrossEntropy(samples_per_class=samples_per_class, beta=cfg.TRAIN.LOSS.BETA, gamma=cfg.TRAIN.LOSS.GAMMA)
    elif cfg.TRAIN.LOSS.NAME == 'cross_entropy':
        return torch.nn.CrossEntropyLoss(reduction=reduction)
    elif cfg.TRAIN.LOSS.NAME == 'focal_softmax_cross_entropy':
        return FocalSoftmaxCrossEntropy(num_classes=cfg.DATA.NUM_CLASSES, gamma=cfg.TRAIN.LOSS.GAMMA)
    else:
        raise RuntimeError('Please check cfg.LOSS.NAME, it must be cbf_softmax_cross_entropy, cross_entropy or focal_softmax_cross_entropy.')


def get_optimizer(cfg, model):
    if cfg.TRAIN.OPTIMIZER.NAME == "Adam": 
        parameters = [param for name, param in model.module.named_parameters() if param.requires_grad]
        # parameters = [
        #     {
        #         'layer_name': 'current_img_encoder', 
        #         'params': [param for name, param in model.module.named_parameters() if 'current' in name],
        #         'lr': config.TRAIN.LR.CUR_IMG_ENCODER 
        #     },
        #     {
        #         'layer_name': 'history_img_encoder',
        #         'params': [param for name, param in model.module.named_parameters() if 'history' in name],
        #         'lr': config.TRAIN.LR.HIS_IMG_ENCODER
        #     },
        #     {
        #         'layer_name': 'text_encoder',
        #         'params': [param for name, param in model.module.named_parameters() if 'text' in name],
        #         'lr': config.TRAIN.LR.TEXT_ENCODER
        #     },
        #     {
        #         'layer_name': 'fusion_part',
        #         'params': [param for name, param in model.module.named_parameters() if 'fusion' in name],
        #         'lr': config.TRAIN.LR.FUSION_PART
        #     },
        #     {
        #         'layer_name': 'classify_layer',
        #         'params': [param for name, param in model.module.named_parameters() if 'classify_layer' in name]
        #     }
        # ]
        # if dist.get_rank() == 0:
        #     print('-' * 40)
        #     print('Learning Rate Setting')
        #     print('-' * 40)
        #     for i in range(len(parameters)):
        #         if 'lr' in parameters[i]:
        #             print('{:^16s} {:^8.6f}' .format(parameters[i]['layer_name'], parameters[i]['lr']))
        #         else:
        #             print('{:^16s} {:^8.6f}' .format(parameters[i]['layer_name'], config.TRAIN.LR.BASE))
        #     print('-' * 40)

        optimizer = torch.optim.Adam(
            params=parameters, lr=cfg.TRAIN.OPTIMIZER.LR, weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        # optimizer = torch.optim.Adam(
        #     model.parameters(), lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    else:
        raise RuntimeError('Please check TRAIN.OPTIMIZER.NAME, it must be Adam.')

    return optimizer


def get_grad_norm(model, norm_type=2):
    """
    裁剪参数迭代的梯度范数
    参考 https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    @param norm_type: 规定正则化范数的类型
    @return: 更新裁剪后的梯度
    """
    parameters = model.parameters
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # 将parameters中的非空网络参数存入一个列表
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0

    # 计算所有网络参数梯度范数之和，再归一化
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


class LoggingUtils:
    '''
    - manage log file.
    '''

    def __init__(self, cfg):
        self.log_dir = cfg.log_dir
        self.log_name = '{}_train.log'.format(datetime.datetime.now().strftime("%Y-%m-%d"))
        self.log_file = os.path.join(self.log_dir, self.log_name)
        # the output format of log
        logging.basicConfig(
            level=logging.INFO,  # CRITICAL > ERROR > WARNING > INFO > DEBUG，default: WARNING
            format='%(asctime)s %(levelname)s:  %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=self.log_file,
            filemode='a'
        )

    @staticmethod
    def info_logger(content):
        logging.info(content)

    @staticmethod
    def error_logger(content):
        logging.error(content)

    @staticmethod
    def critical_logger(content):
        logging.critical(content)


class ClassBalancedFocalSoftmaxCrossEntropy:
    '''class balanced focal softmax cross entropy for classification'''

    def __init__(self, samples_per_class, beta=0.9, gamma=2):
        # beta can be a float, also can be a numpy array whose len is equal to num of classes
        self.beta = beta
        self.gamma = gamma
        self.samples_per_class = samples_per_class
        effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
        class_decay = (1.0 - self.beta) / effective_num
        self.class_decay = torch.from_numpy(class_decay / np.sum(class_decay) * len(self.samples_per_class)).cuda()
        self.num_classes = len(samples_per_class)

    def __call__(self, predict, label):
        # predict: batch size × num_classes
        # label: batch size
        predict = predict.softmax(dim=1)  # batch size × num_classes
        label = label.squeeze()  # batch size
        label_vector = F.one_hot(label, num_classes=self.num_classes)  # batch size × num_classes
        real_proba = torch.sum(predict * label_vector, dim=1) # batch_size
        loss = self.class_decay[label] * torch.pow((1.0 - real_proba), self.gamma) * torch.log(real_proba)
        loss = - torch.sum(loss) / label.shape[0]

        return loss


class FocalSoftmaxCrossEntropy:

    def __init__(self, num_classes, gamma=1.0):
        self.num_classes = num_classes
        self.gamma = gamma

    def __call__(self, predict, label):
        scaled_predict = (predict - torch.max(predict)).softmax(dim=1)
        label_vector = F.one_hot(label, num_classes=self.num_classes)
        real_proba = torch.sum(scaled_predict * label_vector, dim=1)
        loss = torch.pow((1.0 - real_proba), self.gamma) * torch.log(real_proba)
        loss = - torch.sum(loss) / label.shape[0]

        return loss


class LabelSmoothingLoss(_Loss):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing=0, tgt_vocab_size=0, ignore_index=0, size_average=None, reduce=None, reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_vocab_size > 0

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size * num_pos * n_classes
        target (LongTensor): batch_size * num_pos
        """
        assert self.tgt_vocab_size == output.size(2)
        batch_size, num_pos = target.size(0), target.size(1)
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='none').view(batch_size, num_pos, -1).sum(2)
