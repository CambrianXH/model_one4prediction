# 参考资料：
# 1、https://discuss.pytorch.org/t/my-transformer-nmt-model-is-giving-nan-loss-value/122697
# 2、https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

import math
import torch.nn as nn
import torch


class TransForPreTrainingLossMaskPlus(nn.Module):
    def __init__(self, in_features_size=48, out_features_size=2, d_model=512,
                 nhead=8, num_layers=8, dropout=0.1, num_embeddings=1024,
                 tgt_seq_len=50,is_img=False,in_img_size=1792,out_img_size=256):
        super(TransForPreTrainingLossMaskPlus, self).__init__()

        self.src_pos_embedding = torch.nn.Embedding(
            num_embeddings, embedding_dim=d_model)
        self.tgt_pos_embedding = torch.nn.Embedding(
            num_embeddings, embedding_dim=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=4 * d_model,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=4 * d_model,
        )

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers)

        self.is_img = is_img
        self.in_img_size = in_img_size
        if not self.is_img:
            self.in_img_size = 0 
            out_img_size = 0
        else:
            self.fc = nn.Linear(self.in_img_size,out_img_size)
        self.input_projection = nn.Linear(in_features_size + out_img_size, d_model)
        self.output_projection = nn.Linear(out_features_size, d_model)  # 2x768
        self.output_pred = nn.Linear(d_model, out_features_size)   
        self.tgt = nn.Embedding(tgt_seq_len,out_features_size)   # 50X2
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.input_projection.weight.data.uniform_(-init_range, init_range)
        self.output_projection.weight.data.uniform_(-init_range, init_range)
        self.output_pred.bias.data.zero_()
        self.output_pred.weight.data.uniform_(-init_range, init_range)

    def forward(self, seq_txt,seq_img_feature=None):
        src,tgt = seq_txt
        src = self.encode_src(src,seq_img_feature)
        out = self.decode_tgt(tgt = tgt, memory = src)
        return out

    def encode_src(self, src, seq_img_feature):
        if self.is_img:
            img_feature = self.fc(seq_img_feature)
            src = torch.cat([src,img_feature],dim=2) #此处可做多模态融合 by wangyue
        src_embed = self.input_projection(src).permute(1, 0, 2)
        in_seq_len, batch_size = src_embed.size(0), src_embed.size(1)
        pos_encoder = (
            torch.arange(0, in_seq_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        pos_encoder = self.src_pos_embedding(pos_encoder).permute(1, 0, 2)
        src = src_embed + pos_encoder
        src = self.encoder(src)
        return src

    def decode_tgt(self, tgt, memory):
        tgt_embed = self.output_projection(tgt).permute(1, 0, 2)
        out_seq_len, batch_size = tgt_embed.size(0), tgt_embed.size(1)
        tgt_mask = self.gen_tgt_mask(out_seq_len, memory.device)
        out = self.decoder(tgt=tgt_embed, memory=memory,tgt_mask=tgt_mask)
        out = out.permute(1, 0, 2)
        out = self.output_pred(out)

        return out

    def gen_tgt_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)) == 1
        mask = (
            mask.float()
            .masked_fill(mask == 0, float(1.0))
            .masked_fill(mask == 1, float(0.0))
        )
        # 对角线上为-inf，对角线以下是0.0
        return mask



if __name__ == "__main__":
    src = torch.rand(size=(2, 80, 9))
    tgt_in = torch.rand(size=(2, 1, 2))
    input = (src,tgt_in)
    # tgt_out = tgt_in

    model = TransForPreTrainingLossMaskPlus(
        in_features_size=9, out_features_size=2)
    pred = model(input)
    print(pred.size())
    y_hat = pred
    y_hat = y_hat.view(-1)
    criterion = nn.MSELoss()
    loss = criterion(y_hat, pred)

    print("train_loss", loss)
