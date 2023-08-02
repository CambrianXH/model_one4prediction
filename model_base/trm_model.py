import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import math
import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # 64*512
        position = torch.arange(
            0, max_len, dtype=torch.float).unsqueeze(1)  # 64*1
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-math.log(10000.0) / d_model))  # 256   model/2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = nn.Parameter(pe, requires_grad=False)
        #self.register_buffer('pe', pe)   #64*1*512

    def forward(self, x):  # [seq,batch,d_model]
        return x + self.pe[:x.size(0), :]  # 64*64*512


class TransForPreTrainingLossMask(nn.Module):
    def __init__(self, in_features_size=48, out_features_size=2, d_model=512,
                 nhead=8, num_layers=8, dropout=0.1, src_seq_len=80,pad_id=-1000.0):
        super(TransForPreTrainingLossMask, self).__init__()
        self.src_seq_len = src_seq_len
        self.pad_id = pad_id
        self.encoder = nn.Linear(in_features_size, d_model) # shape:[feature_size,512]
        self.src_pos_encoder = PositionalEncoding(
            d_model, max_len = src_seq_len)  # 80*512
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, out_features_size)
        self.init_weights()
        self.src_key_padding_mask = None

    def init_weights(self):
        init_range = 0.1
        # self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src=None):
        #实现attmask padding功能，但是测试增加后，效果好一点点， 如下注释代码先保留，勿动
        #shape of src [seq,batch,feature_size]，
        # if self.src_key_padding_mask is None:
        #     key_padding_mask = torch.ones(
        #         src.shape[1], src.shape[0]).to(src.device)  # [batch,seq]
        #     # mask_key = key_padding_mask.bool()   #[batch,seq]
        #     mask_key = key_padding_mask  # [batch,seq]
        #     self.src_key_padding_mask = mask_key
        
        # 实现src seq 对角线以上为-inf，增加改功能效果不明显
        seq_len = src.size(0)
        # src_mask = self.get_src_mask(src,self.pad_id) # 对pad_id值设置为false做mask
        src_mask = self.get_src_mask(seq_len,src.device) #Encoder 的注意力 Mask 输入，这部分其实对于 Encoder 来说是没有用的，所以这里全是 0
        if seq_len != self.src_seq_len:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]

        src_padding_mask = (src == self.pad_id).to(src.device)
        # 用于 mask 掉 Encoder 的 Token 序列中的 padding 部分,[batch_size, src_len]

        src = self.encoder(src)  # [seq,batch,d_model]
        src = self.src_pos_encoder(src)  # [seq,batch,d_model]
        output = self.transformer_encoder(src,mask = src_mask)# ,src_key_padding_mask = src_padding_mask
        output = self.decoder(output)  # [seq,batch,output_size]
        # self.src_key_padding_mask = None
        return output

    def get_src_mask(self,seq_len, device):
        mask = torch.zeros((seq_len, seq_len), device = device).type(torch.bool)
        return mask

    def get_tgt_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        
        
