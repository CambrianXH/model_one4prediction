import os
import numpy as np
import torch


class TrajEvaluator():
    def __init__(self) :
        self.criterion=torch.nn.MSELoss(reduction="mean")


    def compute_single_ade(self, pred_trajectories, gt_trajectory):
        # pred_trajectories:np.array[N,seq_len,2],gt_trajectory:np.array[seq_len,2]
        # return multi_ade: [N,1]
        # pred_trajectories = np.cumsum(pred_trajectories,axis=1)
        # gt_trajectory = np.cumsum(gt_trajectory,axis=0)
        displacement_errors = np.linalg.norm(pred_trajectories - gt_trajectory, axis=2)  
        multi_ade = np.mean(displacement_errors, axis=1)
        return multi_ade


    def compute_single_fde(self, pred_trajectories, gt_trajectory):
        # pred_trajectories:np.array[N,seq_len,2],gt_trajectory:np.array[seq_len,2]
        # return multi_fde: [N,1]
        # pred_trajectories = np.cumsum(pred_trajectories,axis=1)
        # gt_trajectory = np.cumsum(gt_trajectory,axis=0)
        fde_vector = (pred_trajectories - gt_trajectory)[:, -1] 
        multi_fde = np.linalg.norm(fde_vector, axis=-1)  
        return multi_fde
    
    def compute_prediction_metrics(self,pred_trajs,gt_trajs):
        # pred_trajs:Tensor[B,N,seq_len,2]  , gt_trajs:Tensor[B,seq_len,2]

        # print("begin evaluate trajectory")
        metric = {}
        batch_size,N,seq_len,_ = pred_trajs.shape
        pred_trajs = pred_trajs.cpu().numpy()
        gt_trajs = gt_trajs.cpu().numpy()

        ade,fde,min_ade,min_fde = 0,0,0,0
        for idx in range(batch_size):
            pred_traj,gt_traj = pred_trajs[idx],gt_trajs[idx]
            multi_ade = self.compute_single_ade(pred_traj,gt_traj)
            multi_fde = self.compute_single_fde(pred_traj,gt_traj)

            ade = (ade * idx + sum(multi_ade) / N) / (idx + 1)
            fde = (fde * idx + sum(multi_fde) / N) / (idx + 1)
        
        metric["ADE"] = ade
        metric["FDE"] = fde
        # metric["minADE"] = min_ade / batch_size
        # metric["minFde"] = min_fde / batch_size
        return metric

    def compute_batch_ade(self,pred_trajs,gt_trajs):
        # pred_trajs:Tensor[B,N,seq_len,2]  , gt_trajs:Tensor[B,seq_len,2]

        batch_size,N,seq_len,_ = pred_trajs.shape
        pred_trajs = pred_trajs.cpu().numpy()
        gt_trajs = gt_trajs.cpu().numpy()

        ade_list = []
        for idx in range(batch_size):
            pred_traj,gt_traj = pred_trajs[idx],gt_trajs[idx]
            multi_ade = self.compute_single_ade(pred_traj,gt_traj)
            ade_list.append(multi_ade[0])
        
        return ade_list


    def compute_rmse(self,pred_trajs,gt_trajs):
        '''
        1) 先对预测轨迹值和真值做归一化出来，注：x/y/v/a/theta
        2) 对x/y/v/a/theta分别做差值，再平方后求和再除以N，最后再开方
        '''
        
        pred_mean, pred_std = pred_trajs.mean(dim=1), pred_trajs.std(dim=1) + 1e-10
        gt_mean, gt_std = gt_trajs.mean(dim=1), gt_trajs.std(dim=1) + 1e-10

        norm_pred = (pred_trajs - pred_mean.unsqueeze(1)) / pred_std.unsqueeze(1)
        norm_gt = (gt_trajs - gt_mean.unsqueeze(1))/gt_std.unsqueeze(1)

        rmse = torch.sqrt(self.criterion(norm_pred, norm_gt))

        return rmse.item()



if __name__ == '__main__':

    pred_trajs = torch.arange(12,dtype=torch.float32).reshape(2,3,2)
    gt_trajs = torch.arange(21,27,step=0.5,dtype=torch.float32).reshape(2,3,2)

    evaluator = TrajEvaluator()
    evaluator.compute_rmse(pred_trajs,gt_trajs)
    
