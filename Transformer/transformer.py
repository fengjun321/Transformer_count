#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2021/7/7 19:52
@Author  : Xie Cheng
@File    : transformer.py
@Software: PyCharm
@desc: transformer架构  https://zhuanlan.zhihu.com/p/370481790
"""
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return self.dropout(x)


class TransformerTS(nn.Module):
    def __init__(self,
                 input_dim,
                 dec_seq_len,
                 out_seq_len,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 custom_encoder=None,
                 custom_decoder=None):
        r"""A transformer model. User is able to modify the attributes as needed. The architecture
        is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
        Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
        Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
        Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
        model with corresponding parameters.

        Args:
            input_dim: dimision of imput series
            d_model: the number of expected features in the encoder/decoder inputs (default=512).
            nhead: the number of heads in the multiheadattention models (default=8).
            num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
            num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
            activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
            custom_encoder: custom encoder (default=None).
            custom_decoder: custom decoder (default=None).

        Examples::
            >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
            >>> src = torch.rand((10, 32, 512)) (time length, N, feature dim)
            >>> tgt = torch.rand((20, 32, 512))
            >>> out = transformer_model(src, tgt)

        Note: A full example to apply nn.Transformer module for the word language model is available in
        https://github.com/pytorch/examples/tree/master/word_language_model
        """
        super(TransformerTS, self).__init__()
        self.transform = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder
        )
        self.pos = PositionalEncoding(d_model)
        self.enc_input_fc = nn.Linear(input_dim, d_model)
        self.dec_input_fc = nn.Linear(input_dim, d_model)
        self.out_fc = nn.Linear(dec_seq_len * d_model, out_seq_len)
        self.dec_seq_len = dec_seq_len

    def forward(self, x):
        x = x.transpose(0, 1)
        # embedding
        embed_encoder_input = self.pos(self.enc_input_fc(x))
        embed_decoder_input = self.dec_input_fc(x[-self.dec_seq_len:, :])
        # transform
        x = self.transform(embed_encoder_input, embed_decoder_input)

        # output
        x = x.transpose(0, 1)
        x = self.out_fc(x.flatten(start_dim=1))
        return x.squeeze()
