from Transformer import *

class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        nn.init.xavier_uniform(self.weight)
        #nn.init.normal_(self.weight, mean=0, std=np.sqrt(2.0 / (n_in)))
        #self.reset_parameters()

    def extra_repr(self):
        info = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            info += f", bias_x={self.bias_x}"
        if self.bias_y:
            info += f", bias_y={self.bias_y}"

        return info

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat([x, x.new_ones(x.shape[:-1]).unsqueeze(-1)], -1)
        if self.bias_y:
            y = torch.cat([y, y.new_ones(y.shape[:-1]).unsqueeze(-1)], -1)
        # [batch_size, 1, seq_len, d]
        x = x.unsqueeze(1)
        # [batch_size, 1, seq_len, d]
        y = y.unsqueeze(1)
        # [batch_size, n_out, seq_len, seq_len]
        s = x @ self.weight @ y.transpose(-1, -2)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s

class ScaledDotProductAttention_composition(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, d_k, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

        '''self.U=nn.Linear(d_k, d_k)
        self.W=nn.Linear(d_k, d_k)
        nn.init.normal_(self.U.weight, mean=0, std=np.sqrt(2.0 / (d_k)))
        nn.init.normal_(self.W.weight, mean=0, std=np.sqrt(2.0 / (d_k)))'''
        self.composition_attn=Biaffine(n_in=d_k,
                                 bias_x=True,
                                 bias_y=False)

    def forward(self, q, k, v, mask=None):

        #attn = torch.bmm(q, k.transpose(1, 2))
        #attn = attn / self.temperature

        attn = self.composition_attn(q,k)

        if mask is not None:
            attn = attn.masked_fill(mask.byte(), -1e15)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention_composition(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1,layer_index=1.0):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k*2)
        self.w_ks = nn.Linear(d_model, n_head * d_k*2)
        self.w_vs = nn.Linear(d_model, n_head * d_v*2)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention=ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.attention_composition = ScaledDotProductAttention_composition(temperature=np.power(d_k, 0.5),d_k=d_k)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc1 = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc2.weight)

        self.fc = nn.Linear(n_head * d_v*2, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None, edge=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q0 = self.w_qs(q).view(sz_b, len_q, n_head*2, d_k)
        k0 = self.w_ks(k).view(sz_b, len_k, n_head*2, d_k)
        v0 = self.w_vs(v).view(sz_b, len_v, n_head*2, d_v)

        q1,q2=q0[:,:,:self.n_head,:],q0[:,:,self.n_head:,:]
        k1, k2 = k0[:, :, :self.n_head, :], k0[:, :, self.n_head:, :]
        v1, v2 = v0[:, :, :self.n_head, :], v0[:, :, self.n_head:, :]

        q1 = q1.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k1 = k1.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v1 = v1.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        q2 = q2.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k2 = k2.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v2 = v2.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv


        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output1, attn = self.attention(q1, k1, v1, mask=mask)
        output1 = output1.view(n_head, sz_b, len_q, d_v)
        output1 = output1.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        #output1 = self.dropout(self.fc1(output1))

        output2, _ = self.attention_composition(q2, k2, v2, mask=mask)
        output2 = output2.view(n_head, sz_b, len_q, d_v)
        output2 = output2.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        #output2 = self.dropout(self.fc2(output2))

        output = self.dropout(self.fc(torch.cat((output1,output2),-1)))
        output = self.layer_norm(output+ residual)

        return output, attn

class TransformerEncoderLayer_composition(TransformerEncoderLayer):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention_composition(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

class Encoder_composition(Encoder):
    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__(n_src_vocab, len_max_seq, d_word_vec,n_layers, n_head, d_k, d_v,d_model, d_inner, dropout)

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=0)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            TransformerEncoderLayer_composition(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

class Transofmer_compositional(Transformer):
    def __init__(
            self,
            n_src_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1
    ):
        super().__init__(n_src_vocab, len_max_seq,
            d_word_vec, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout)

        self.encoder = Encoder_composition(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)
        self.A_weight = nn.Parameter(torch.ones(d_model, 1))
        self.d_model=d_model
        nn.init.xavier_normal_(self.A_weight)