import torch
import torch.nn as nn
import torch.nn.functional as F
from Config_File import Config
import numpy as np

def ProjectPlane(vec,plane):
    return vec- torch.sum(vec*plane,-1,keepdim=True)/(torch.norm(plane,p=2,dim=-1,keepdim=True)+1e-8)*plane

def ComputeRelativeCat(emb1, emb2):
    sim = torch.norm(emb1.unsqueeze(-2)-emb2.unsqueeze(-3),p=2,keepdim=False,dim=-1)
    _, sim_max_index = torch.min(sim, -1)
    relative_emb = torch.gather(emb2, -2, sim_max_index.unsqueeze(-1).expand(-1, -1, emb1.size(-1)))

    return torch.cat((emb1, emb1 - relative_emb), -1)
    #return emb1- relative_emb
    #return emb1*(emb1 - relative_emb)

def ComputeRelative(emb1, emb2):
    sim = torch.norm(emb1.unsqueeze(-2)-emb2.unsqueeze(-3),p=2,keepdim=False,dim=-1)
    _, sim_max_index = torch.min(sim, -1)
    relative_emb = torch.gather(emb2, -2, sim_max_index.unsqueeze(-1).expand(-1, -1, emb1.size(-1)))

    return emb1-relative_emb

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1,non_linear=False):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.non_linear=non_linear
        self.sigma=F.sigmoid

    def forward(self, q, k, v, mask=None, edge=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature


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

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1,layer_index=1.0):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None, edge=None):

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

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask, edge=edge)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class TransformerEncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None, edge=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask, edge=edge)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        if not Config.use_pre_train:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn