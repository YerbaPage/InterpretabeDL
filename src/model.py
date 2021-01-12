import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Transformer import *
from Config_File import Config

class BiLSTM2(nn.Module):
    def __init__(self, word_embedding_dim, rnn_dim, dropout):
        super(BiLSTM2, self).__init__()
        self.input_dim = word_embedding_dim
        self.output_dim = rnn_dim
        self.rnns = nn.ModuleList(
            [nn.LSTM(input_size=self.input_dim, hidden_size=self.output_dim // 2, batch_first=True,
                     bidirectional=True)] +
            [nn.LSTM(input_size=self.output_dim * 2, hidden_size=self.output_dim, batch_first=True, bidirectional=True)
             for i in range(0)])
        self.d_model = rnn_dim

    def forward(self, states, mask=None):
        outputs = []
        if mask is not None:
            mask = torch.unsqueeze(mask, dim=2).expand(mask.size(0), mask.size(1), self.output_dim)
        for rnn in self.rnns:
            output, (h_n, c_n) = rnn(states)
            if mask is not None:
                output = output * mask.float()
            states = output
            outputs.append(output)

        infi_mask = mask.lt(0.5).float() * (-10e7)
        masked_output = outputs[-1] + infi_mask
        return output,masked_output.max(dim=1)[0]

class BiLSTM(nn.Module):
    def __init__(self, word_embedding_dim, rnn_dim, dropout):
        super(BiLSTM, self).__init__()
        self.input_dim = word_embedding_dim
        self.output_dim = rnn_dim
        self.rnns = nn.ModuleList(
            [nn.LSTM(input_size=self.input_dim, hidden_size=self.output_dim//2, batch_first=True, bidirectional=True)] +
            [nn.LSTM(input_size=self.output_dim*2, hidden_size=self.output_dim, batch_first=True, bidirectional=True)
             for i in range(0)])
        self.d_model=rnn_dim

    def forward(self, states, mask=None):
        outputs = []
        if mask is not None:
            mask = torch.unsqueeze(mask, dim=2).expand(mask.size(0), mask.size(1), self.output_dim)
        for rnn in self.rnns:
            output, (h_n, c_n) = rnn(states)
            if mask is not None:
                #output = output * mask.float()
                output = output.masked_fill(mask.eq(0), 0)
            states = output
            outputs.append(output)

        masked_output=outputs[-1]
        if mask is not None:
            masked_output-=(1-mask.float())*1e7
            output_maxpooled=masked_output.max(dim=1)[0]
            output_maxpooled=torch.max(output_maxpooled,torch.ones(output_maxpooled.size()).to(Config.device)*-1)
        return outputs[-1],output_maxpooled


class ComputeTwoWordEmb(nn.Module):
    def forward(self,emb,mask,pos):
        emb=emb[:,:30,:]
        mask=mask[:,:30]
        pos=pos[:,:30]
        length=emb.size()[-2]
        emb1=emb.unsqueeze(-2).expand(-1,-1,length,-1)
        mask1=mask.unsqueeze(-1).expand(-1,-1,length)
        pos1 = pos.unsqueeze(-1).expand(-1, -1, length)
        emb2=emb.unsqueeze(-3).expand(-1,length,-1,-1)
        mask2 = mask.unsqueeze(-2).expand(-1, length,-1)
        mask=(mask1+mask2).gt(1).view(emb.size()[0],-1)

        return ((emb1+emb2)/2).view(emb.size()[0],-1,emb.size()[2]),mask,pos1.contiguous().view(emb.size()[0],-1)

class ComputeEdgeWordEmb(nn.Module):
    def forward(self,emb,pos,edge):
        edge_half=edge[:,:,:Config.sent_len].float()+1e-8
        edge_half=edge_half/torch.sum(edge_half,dim=-1,keepdim=True)
        emb1=torch.bmm(edge_half,emb)
        pos=torch.cat((pos,pos),dim=-1)
        return emb1,pos,edge

class ComputeEdgeChunkWordEmb(nn.Module):
    def forward(self,emb,pos,edge):
        edge_half=edge[:,:,:Config.sent_len].float()+1e-8
        edge_half=edge_half/torch.sum(edge_half,dim=-1,keepdim=True)
        emb1=torch.bmm(edge_half,emb)
        pos=torch.cat((pos,pos,pos),dim=-1)
        return emb1,pos,edge

class ComputeTwoNeighborWordEmb(nn.Module):
    def __init__(self, lengthk, k=2):
        # edge is used for self attention in lm. adj is used for inductive bias
        super(ComputeTwoNeighborWordEmb, self).__init__()
        self.adj=torch.eye(lengthk).unsqueeze(0).to(Config.device)
        length=int(lengthk / k)
        self.k=k
        for k1 in range(1,k):
            for k2 in range(k1+1,k+1):
                for i in range((k1-1)*length,k1*length):
                    for j in range((k2-1)*length,k2*length):
                        li=max(0, i - (k1-1) * length + k1 // 2 - (k1 - 1))
                        ri=min(i - (k1-1) * length + k1 // 2, length - 1)
                        lj = max(0, j - (k2-1) * length + k2 // 2 - (k2 - 1))
                        rj = min(j - (k2-1) * length + k2 // 2, length - 1)

                        if lj<=li and ri<=rj:
                            self.adj[0,i,j]=1
                            self.adj[0,j,i]=1
        if '_lm' in Config.dataset:
            self.edge=torch.zeros((lengthk, lengthk)).unsqueeze(0).to(Config.device)
            for k1 in range(1,k+1):
                for i1 in range(length):
                    for k2 in range(k):
                        for i2 in range(length):
                            if min(i1+k1//2,length-1)>=min(i2+k2//2,length-1):
                                self.edge[0][k1*length+i1][k2*length+i2]=1
            self.adj=self.adj*self.edge

    def forward(self,emb,mask,pos):
        emb_list=[emb]
        pos_list=[pos]
        if self.k>=2:
            for k in range(2,self.k+1):
                embk = []
                for pad_left in range(k):
                    embk.append(F.pad(emb,(0,0,pad_left,k-1-pad_left,0,0),'constant',0))
                #embk=torch.sum(torch.stack(embk,0),0,keepdim=False)/k
                embk= torch.mean(torch.stack(embk),dim=0)
                embk=embk[:,k//2:k//2+Config.sent_len,:]
                emb_list.append(embk)
                pos_list.append(pos)

        if '_lm' not in Config.dataset:
            return torch.cat(emb_list, 1), None, torch.cat(pos_list, 1), self.adj
        else:
            return torch.cat(emb_list, 1), self.edge, torch.cat(pos_list, 1), self.adj

class ComputeLMMask(nn.Module):
    def forward(self,len_src,mask):
        if '_lm' in Config.dataset:
            mask = (torch.triu(torch.ones(len_src, len_src)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, 0).masked_fill(mask == 1, float(1))
            mask = mask.unsqueeze(0).to(Config.device)
        else:
            mask=None
        return mask

class ExtractTopN(nn.Module):
    def __init__(self,d_model):
        super(ExtractTopN, self).__init__()
        self.d_model=d_model
    def forward(self,emb,mask,pos,n):
        return emb[:,:n,:],mask[:,:n],pos[:,:n]

import math
class ExtractEye(nn.Module):
    def __init__(self,d_model,seq_len):
        super(ExtractEye, self).__init__()
        self.d_model=d_model
        self.seq_len=seq_len
    def forward(self,emb):
        length2=emb.size()[1]
        length=int(math.sqrt(length2))
        emb=emb.view(emb.size()[0],length,length,-1)
        emb=emb.diagonal(dim1=-3, dim2=-2).transpose(1,2)
        pad_size=(0,0,0,self.seq_len-length,0,0)
        emb=F.pad(emb,pad_size,"constant", 0)
        return emb,emb[:,0,:]

class Regress(nn.Module):
    def __init__(self, dim):
        super(Regress, self).__init__()
        self.A_weight = nn.Parameter(torch.ones(dim, 1)).to(Config.device)
        nn.init.xavier_normal_(self.A_weight)
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, sent_embeded):
        sent_embeded = self.dropout_layer(sent_embeded)
        out = torch.mm(sent_embeded, self.A_weight)

        return torch.sigmoid(out)


class Classifier(nn.Module):
    def __init__(self, dim, class_num,dropout=0.2):
        super(Classifier, self).__init__()
        self.linear=nn.Linear(dim, class_num)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, sent_embeded):
        sent_embeded = self.dropout_layer(sent_embeded)
        out=self.linear(sent_embeded)

        return out

class BiLSTM_pair(nn.Module):
    def __init__(self, embeding_dim, rnn_dim, dropout):
        super(BiLSTM_pair, self).__init__()
        self.BiLSTM1 = BiLSTM(embeding_dim, rnn_dim, dropout)
        #self.BiLSTM2 = BiLSTM(embeding_dim, rnn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, return_sequence=False, overlap_mask1=None, overlap_mask2=None):
        sent, sent_mask, sent2, sent_mask2=inputs
        sent_embeded1=self.BiLSTM1(sent,sent_mask,return_sequence)
        sent_embeded2 = self.BiLSTM1(sent2, sent_mask2, return_sequence)
        #sent_embeded1 = self.dropout(sent_embeded1)
        #sent_embeded2 = self.dropout(sent_embeded2)

        '''if overlap_mask1 is not None:
            sent_embeded1 = sent_embeded1 * overlap_mask1.unsqueeze(-1).expand(-1,-1,sent_embeded1.size()[-1])
            sent_embeded2 = sent_embeded2 * overlap_mask2.unsqueeze(-1).expand(-1,-1,sent_embeded2.size()[-1])'''

        merged_embed = torch.cat((sent_embeded1, sent_embeded2, torch.abs(sent_embeded1 - sent_embeded2), sent_embeded1 * sent_embeded2),dim=-1)
        #merged_embed=torch.cat((sent_embeded1,sent_embeded2,sent_embeded1 * sent_embeded2),dim=-1)
        #merged_embed = torch.cat((sent_embeded1, sent_embeded2), dim=-1)

        return self.dropout(merged_embed)

class WordEmbedding(nn.Module):
    def __init__(self,vocab_size,dim,weights_matrix):
        super(WordEmbedding, self).__init__()
        self.embed_layer=nn.Embedding(vocab_size, dim).from_pretrained(torch.from_numpy(weights_matrix).float(),freeze=False)
        #self.embed_layer.weight=nn.Parameter(torch.from_numpy(weights_matrix), requires_grad=True)
        #self.embed_layer.load_state_dict({'weight': weights_matrix})

    def forward(self, input, mask=None):
        output=self.embed_layer(input)
        if mask is not None:
            output=output*mask.float().unsqueeze(-1).expand(-1,-1,output.size()[-1])
        return output

class WordPairMatch(nn.Module):
    def __init__(self,rnn_dim,dropout):
        super(WordPairMatch, self).__init__()
        self.embed_layer = nn.Embedding(3, Config.word_dim)
        self.BiLSTM1 = BiLSTM(Config.word_dim, rnn_dim, dropout)
        self.BiLSTM2 = BiLSTM(Config.word_dim, rnn_dim, dropout)

    def sent_pair_match(self, sents1, sents2):
        sents1=sents1.cpu().numpy()
        sents2=sents2.cpu().numpy()
        ret1=np.zeros(sents1.shape)
        ret2 = np.zeros(sents2.shape)
        for i, sent in enumerate(sents1):
            for j,t in enumerate(sents1[i]):
                if t==0:
                    ret1[i,j]=0
                elif t in sents2[i]:
                    ret1[i,j]=1
                else:
                    ret1[i,j]=2
        for i, sent in enumerate(sents2):
            for j,t in enumerate(sents2[i]):
                if t==0:
                    ret2[i,j]=0
                elif t in sents1[i]:
                    ret2[i,j]=1
                else:
                    ret2[i,j]=2
        return torch.LongTensor(ret1).to(Config.device),torch.LongTensor(ret2).to(Config.device)

    def forward(self, inputs):
        sent, sent_mask, sent2, sent_mask2=inputs
        sent,sent2=self.sent_pair_match(sent,sent2)
        sent=self.embed_layer(sent)
        sent2=self.embed_layer(sent2)
        sent_embeded1=self.BiLSTM1(sent,sent_mask)
        sent_embeded2 = self.BiLSTM2(sent2, sent_mask2)
        merged_embed=torch.cat((sent_embeded1,sent_embeded2,torch.abs(sent_embeded1-sent_embeded2),sent_embeded1*sent_embeded2),dim=-1)
        #merged_embed = torch.cat((sent_embeded1, sent_embeded2,sent_embeded1 * sent_embeded2), dim=-1)
        #merged_embed = torch.cat((sent_embeded1, sent_embeded2), dim=-1)

        return merged_embed

class WordPairMatch_HideOverlap(nn.Module):
    def __init__(self,weights_matrix,dropout):
        super(WordPairMatch_HideOverlap, self).__init__()
        self.embed_layer = nn.Embedding(len(weights_matrix), Config.word_dim).from_pretrained(torch.from_numpy(weights_matrix).float(),
                                                                         freeze=False)
        self.BiLSTM1 = BiLSTM(Config.word_dim, Config.rnn_dim, dropout)
        self.BiLSTM2 = BiLSTM(Config.word_dim, Config.rnn_dim, dropout)

    def sent_pair_match(self, sents1, sents2):
        sents1=sents1.cpu().detach().numpy()
        sents2=sents2.cpu().detach().numpy()
        ret1=np.zeros(sents1.shape)
        ret2 = np.zeros(sents2.shape)
        for i, sent in enumerate(sents1):
            for j,t in enumerate(sents1[i]):
                if t==0:
                    ret1[i,j]=0
                elif t in sents2[i]:
                    ret1[i,j]=1
                else:
                    ret1[i,j]=t
        for i, sent in enumerate(sents2):
            for j,t in enumerate(sents2[i]):
                if t==0:
                    ret2[i,j]=0
                elif t in sents1[i]:
                    ret2[i,j]=1
                else:
                    ret2[i,j]=t
        return torch.LongTensor(ret1).to(Config.device),torch.LongTensor(ret2).to(Config.device)

    def forward(self, inputs):
        sent, sent_mask, sent2, sent_mask2=inputs
        sent,sent2=self.sent_pair_match(sent,sent2)
        sent=self.embed_layer(sent)
        sent2=self.embed_layer(sent2)
        sent_embeded1=self.BiLSTM1(sent,sent_mask)
        sent_embeded2 = self.BiLSTM2(sent2, sent_mask2)
        merged_embed=torch.cat((sent_embeded1,sent_embeded2,torch.abs(sent_embeded1-sent_embeded2),sent_embeded1*sent_embeded2),dim=-1)
        #merged_embed = torch.cat((sent_embeded1, sent_embeded2,sent_embeded1 * sent_embeded2), dim=-1)
        #merged_embed = torch.cat((sent_embeded1, sent_embeded2), dim=-1)

        return merged_embed

class WordPairMatch_HideOverlap_Positional(WordPairMatch_HideOverlap):
    def sent_pair_match(self, sents1, sents2):
        sents1=sents1.cpu().detach().numpy()
        sents2=sents2.cpu().detach().numpy()
        ret1=np.zeros(sents1.shape)
        ret2 = np.zeros(sents2.shape)
        for i in range(len(sents1)):
            overlap_list=[]
            for t in sents1[i]:
                if t!=0 and (t in sents2[i]) and (t not in overlap_list):
                    overlap_list.append(t)

            for j,t in enumerate(sents1[i]):
                if t==0:
                    ret1[i,j]=0
                elif t in sents2[i]:
                    ret1[i,j]=overlap_list.index(t)
                else:
                    ret1[i,j]=t

            for j,t in enumerate(sents2[i]):
                if t==0:
                    ret2[i,j]=0
                elif t in sents1[i]:
                    ret2[i,j]=overlap_list.index(t)
                else:
                    ret2[i,j]=t
        return torch.LongTensor(ret1).to(Config.device),torch.LongTensor(ret2).to(Config.device)

class WordPairMatch_HideUnoverlap(nn.Module):
    def __init__(self,weights_matrix,dropout):
        super(WordPairMatch_HideUnoverlap, self).__init__()
        self.embed_layer = nn.Embedding(len(weights_matrix), Config.word_dim).from_pretrained(torch.from_numpy(weights_matrix).float(),
                                                                         freeze=False)
        self.BiLSTM1 = BiLSTM(Config.word_dim, Config.rnn_dim, dropout)
        self.BiLSTM2 = BiLSTM(Config.word_dim, Config.rnn_dim, dropout)

    def sent_pair_match(self, sents1, sents2):
        sents1=sents1.cpu().detach().numpy()
        sents2=sents2.cpu().detach().numpy()
        ret1=np.zeros(sents1.shape)
        ret2 = np.zeros(sents2.shape)
        for i, sent in enumerate(sents1):
            for j,t in enumerate(sents1[i]):
                if t==0:
                    ret1[i,j]=0
                elif t in sents2[i]:
                    ret1[i,j]=t
                else:
                    ret1[i,j]=1
        for i, sent in enumerate(sents2):
            for j,t in enumerate(sents2[i]):
                if t==0:
                    ret2[i,j]=0
                elif t in sents1[i]:
                    ret2[i,j]=t
                else:
                    ret2[i,j]=1

        '''for i, sent in enumerate(sents1):
            l=0
            for j, t in enumerate(sents1[i]):
                if t in sents2[i] and t!=0:
                    ret1[i, l] = t
                    l+=1

        for i, sent in enumerate(sents2):
            l=0
            for j, t in enumerate(sents2[i]):
                if t in sents1[i] and t!=0:
                    ret2[i, l] = t
                    l+=1'''

        return torch.LongTensor(ret1).to(Config.device),torch.LongTensor(ret2).to(Config.device)

    def forward(self, inputs):
        sent, sent_mask, sent2, sent_mask2=inputs
        sent,sent2=self.sent_pair_match(sent,sent2)
        sent=self.embed_layer(sent)
        sent2=self.embed_layer(sent2)
        sent_embeded1=self.BiLSTM1(sent,sent_mask)
        sent_embeded2 = self.BiLSTM2(sent2, sent_mask2)
        merged_embed=torch.cat((sent_embeded1,sent_embeded2,torch.abs(sent_embeded1-sent_embeded2),sent_embeded1*sent_embeded2),dim=-1)
        #merged_embed = torch.cat((sent_embeded1, sent_embeded2,sent_embeded1 * sent_embeded2), dim=-1)
        #merged_embed = torch.cat((sent_embeded1, sent_embeded2), dim=-1)

        return merged_embed

class MLP(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # x.mm(w1)+b1
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.Sigmoid()

    def forward(self, x):  #n *input_dim
        x = F.relu(self.fc1(x))  #method a
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x