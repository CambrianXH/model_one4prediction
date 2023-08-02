'''
通用的文本编码模块
Author: wangyue
Date: 2022-10-18 14:57:50
LastEditors: aiwenjie aiwenjie20@outlook.com
LastEditTime: 2022-11-07 14:35:44
FilePath: /wangyue/model_one/model_base/multi_model_encoder.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chnl=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chnl, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class ImgEncoder():
    def __init__(self, img_size=224, patch_size=16, in_chnl=3, embed_dim=768, dropout_ratio=0.5, embed_layer=PatchEmbed):
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chnl=in_chnl, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_ratio)

    def forward(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        x = self.pos_drop(x + self.pos_embed)
        return x




class TxtEncoder():

    """
    对自车、障碍物、车道线等数值信息编码
    类别值要先进行one_hot编码

    """

    def __init__(self, embed_dim=768, lane_nums=2, obst_nums=1):
        self.embed_dim = embed_dim
        self.lane_nums = lane_nums
        self.obst_nums = obst_nums

        self.ego_vec = nn.Parameter(torch.zeros(1, embed_dim))
        self.lane_vec_list = nn.Parameternn.Parameter(
            torch.zeros(lane_nums, embed_dim))
        self.obst_vec_list = nn.Parameter(torch.zeros(obst_nums, embed_dim))

    def forward(self, txt_dict):
        """
        todo:处理字典逻辑
        ego_dict:自车信息字典
        lane_list_dict：车道线信息，是数组格式，内部是字典格式
        obst_list_dict：障碍物信息，是数组格式，内部是字典格式
        """
        txt_vec = []
        # return self.ego_vec,self.lane_vec_list,self.obst_vec_list
        return txt_vec
