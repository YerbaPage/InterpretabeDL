from Transformer import Transformer, Encoder, TransformerEncoderLayer, get_sinusoid_encoding_table, \
    get_attn_key_pad_mask, \
    get_non_pad_mask
import torch.nn as nn
import torch
from modules import MultiHeadAttention, PositionwiseFeedForward, ScaledDotProductAttention, ProjectPlane, \
    ComputeRelative, ComputeRelativeCat
import  numpy as np
import torch.nn.functional as F
from Config_File import Config



class ScaledDotProductAttention_relative(nn.Module): #works
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1,non_linear=False, d_model=Config.inner_dim):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.non_linear=non_linear
        self.sigma=F.sigmoid

        self.fc=nn.Linear(d_model,d_model)
        nn.init.normal_(self.fc.weight, mean=0, std=np.sqrt(2.0 / (d_model+d_model)))
        self.sigma=torch.sigmoid
        self.fc2=nn.Linear(d_model,1)


    def forward(self, q, k, v, k2, mask=None, edge=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        '''q_cat=ComputeRelativeCat(q,k2)
        k_cat=ComputeRelativeCat(k,k2)
        attn = torch.bmm(q_cat,k_cat.transpose(1,2))
        attn = attn / (self.temperature*1.41421)'''

        '''#q, k = ComputeRelative(q,k2), ComputeRelative(k,k2)
        rel=ComputeRelative(k,k2)
        attn_rel=self.sigma(self.fc(rel))
        attn_rel=self.fc2(attn_rel).transpose(1,2)
        #attn_rel=(torch.sum(rel,-1,keepdim=False).eq(0)*-1e8).unsqueeze(-2)
        attn+=attn_rel'''

        if edge is not None:
            edge=edge.repeat(attn.shape[0]//edge.shape[0],1,1).float()#.unsqueeze(-1).float()
            #attn = attn*edge  #should be debug cuiwanyun
            attn = attn.masked_fill(edge.eq(0), -1e15)

        if mask is not None:
            attn = attn.masked_fill(mask, -1e15)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        if self.non_linear:
            output=self.sigma(output)

        return output, attn

class MultiHeadAttention_relative_k(MultiHeadAttention):  #works
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1,layer_index=1.0):
        super(MultiHeadAttention_relative_k,self).__init__(n_head, d_model, d_k, d_v, dropout,layer_index)
        self.attention = ScaledDotProductAttention_relative(temperature=np.power(d_k, 0.5))
        #self.w_ks = nn.Linear(d_model*2, n_head * d_k)
        #nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model*2 + d_k)))

        #self.w_qs = nn.Linear(d_model*2, n_head * d_k)
        #nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model*2 + d_k)))

    def forward(self, q, k, v, k2, mask=None, edge=None, mask2=None, edge2=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        k2 = self.w_ks(k2).view(sz_b, len_k, n_head, d_k)
        k2= k2.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        #k_relative = ComputeRelative(k, k2)


        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, k2, mask=mask, edge=edge)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class MultiHeadAttention_relative(MultiHeadAttention):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, layer_index=1.0):
        super().__init__(n_head, d_model , d_k, d_v, dropout, layer_index) #cwy
        self.fc2 = nn.Linear(d_model * 2, d_model) #cwy
        nn.init.xavier_normal_(self.fc2.weight)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None, edge=None):


        output, attn = super(MultiHeadAttention_relative, self).forward(q, k, v, mask, edge)
        # output = self.dropout(self.fc2(output))  #cwy
        # q_dim_ori=q.size(2)//2
        # output = self.layer_norm2(output +q[:,:,:q_dim_ori] )        #cwy, residual helps
        return output, attn


class TransformerEncoderLayer_relative(TransformerEncoderLayer):  #works
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(TransformerEncoderLayer_relative, self).__init__(d_model, d_inner, n_head, d_k, d_v, dropout)
        self.slf_attn = MultiHeadAttention_relative_k(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.fc = nn.Linear(d_model * 2, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, enc_input, enc_input2, non_pad_mask=None, slf_attn_mask=None, edge=None, non_pad_mask2=None,
                slf_attn_mask2=None, edge2=None):
        '''rel = ComputeMinRelative(enc_input, enc_input2)
        rel2 = ComputeMinRelative(enc_input2, enc_input)
        rel_mask=torch.sum(rel,-1,keepdim=True).ne(0)
        rel_mask2 = torch.sum(rel2,-1,keepdim=True).ne(0)
        non_pad_mask*=rel_mask
        non_pad_mask2 *= rel_mask2'''


        #enc_input = self.fc(enc_input)
        #enc_input2 = self.fc(enc_input2)

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, enc_input2, mask=slf_attn_mask, edge=edge)
        enc_output *= non_pad_mask
        enc_output = self.pos_ffn(enc_output)

        enc_output2, enc_slf_attn2 = self.slf_attn(
            enc_input2, enc_input2, enc_input2, enc_input, mask=slf_attn_mask2, edge=edge2)
        enc_output2 *= non_pad_mask2
        enc_output2 = self.pos_ffn(enc_output2)

        return enc_output, enc_slf_attn, enc_output2, enc_slf_attn2


class Encoder_relative(Encoder):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super(Encoder_relative, self).__init__(n_src_vocab, len_max_seq, d_word_vec, n_layers, n_head, d_k, d_v,
                                               d_model,
                                               d_inner, dropout)

        '''self.layer_stack = nn.ModuleList([
            TransformerEncoderLayer_relative(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) if i==0 else
            TransformerEncoderLayer(d_model*2, d_inner, n_head, d_k, d_v, dropout=dropout)
            for i in range(n_layers)])'''
        self.layer_stack = nn.ModuleList([
            TransformerEncoderLayer_relative(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for i in range(n_layers)])  # cwy

    def forward(self, src_seq, src_embed, src_pos, src_seq2, src_embed2, src_pos2, return_attns=False):

        enc_slf_attn_list = []
        enc_slf_attn_list2 = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_pos, seq_q=src_pos)
        non_pad_mask = get_non_pad_mask(src_pos)
        slf_attn_mask2 = get_attn_key_pad_mask(seq_k=src_pos2, seq_q=src_pos2)
        non_pad_mask2 = get_non_pad_mask(src_pos2)

        # -- Forward
        enc_output = src_embed + self.position_enc(src_pos)
        enc_output2 = src_embed2 + self.position_enc(src_pos2)

        for i, enc_layer in enumerate(self.layer_stack):
            if True:
                enc_output, enc_slf_attn, enc_output2, enc_slf_attn2 = enc_layer(
                    enc_output, enc_output2,
                    non_pad_mask=non_pad_mask,
                    slf_attn_mask=slf_attn_mask,
                    non_pad_mask2=non_pad_mask2,
                    slf_attn_mask2=slf_attn_mask2
                )
            else:
                enc_output, enc_slf_attn = enc_layer(enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
                enc_output2, enc_slf_attn2 = enc_layer(enc_output2, non_pad_mask=non_pad_mask2,
                                                       slf_attn_mask=slf_attn_mask2)

            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
                enc_slf_attn_list2 += [enc_slf_attn2]

        if return_attns:
            return enc_output, enc_slf_attn_list, enc_output2, enc_slf_attn_list2
        return enc_output, enc_output2


class Transformer_relative(Transformer):
    def __init__(
            self,
            n_src_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, non_linear=False):
        super(Transformer_relative, self).__init__(n_src_vocab, len_max_seq, d_word_vec, d_model, d_inner, n_layers,
                                                   n_head,
                                                   d_k, d_v, dropout, non_linear)

        self.encoder = Encoder_relative(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)
        for encoder_layer in self.encoder.layer_stack:
            encoder_layer.slf_attn.attention.non_linear = non_linear
        self.d_model = d_model

    def forward(self, src_seq, src_embed, src_pos, src_seq2, src_embed2,
                src_pos2):  # embedding: batch*len*dim, pos: batch*len

        enc_output, enc_output2, *_ = self.encoder(src_seq, src_embed, src_pos, src_seq2, src_embed2, src_pos2)
        size = enc_output.size()

        return enc_output, enc_output[:, 0, :].squeeze(), enc_output2, enc_output2[:, 0, :].squeeze()
