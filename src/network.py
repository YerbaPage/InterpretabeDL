import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Transformer import *
from model import *


# from crf import CRF
# from omnt_transformer import Omnt_Transformer

class BigModel(nn.Module):

    def __init__(self,args):
        super(BigModel, self).__init__()

        self.pre_emb_model = args.pre_trained_model#[0].from_pretrained(args.pre_trained_model[2],config=args.pre_train_config)

        word_dim = args.config.hidden_size #self.pre_emb_model.embeddings.word_embeddings.embedding_dim
        self.classifier = Classifier(word_dim, args.class_num, dropout=0)

        self.dropout = nn.Dropout(args.dropout)

    #def RepresentEmb(self, input, input_mask, input_pos, input_typeid=None):
    #    return self.pre_emb_model.embeddings.word_embeddings(input)

    def RepresentLongterm(self, input_word_embeding, input_mask=None, input_ori=None, input_edge=None, root=None,
                          pos=None):
        token_type_ids=input_ori['x_typeid'] if 'x_typeid' in input_ori else None

        seq_repre, deep_repre, hidden_repre = self.pre_emb_model(attention_mask=input_ori['x_mask'],
                                                                 token_type_ids=token_type_ids,
                                                                 position_ids=input_ori['x_pos'],
                                                                 inputs_embeds=input_word_embeding)

        return deep_repre, seq_repre, hidden_repre

    def ReprentOneSent(self, input, input_mask=None, input_edge=None, input_edge_root=None, input_pos=None,
                       input_typeid=None):
        input_word_embeding, extended_attention_mask, head_mask, encoder_hidden_states, encoder_extended_attention_mask = self.RepresentEmb(
            input, input_mask, input_pos, input_typeid=input_typeid)
        return self.RepresentLongterm(input_word_embeding=input_word_embeding, input_mask=input_mask,
                                      input_ori=input,
                                      input_edge=input_edge,
                                      root=input_edge_root, pos=input_pos,
                                      extended_attention_mask=extended_attention_mask, head_mask=head_mask,
                                      encoder_hidden_states=encoder_hidden_states,
                                      encoder_extended_attention_mask=encoder_extended_attention_mask)

    def forward(self, input, return_hidden_states=False):
        #temp = self.forward_emb(input)
        #return self.forward_long(temp, input, return_hidden_states)

        outputs= self.pre_emb_model(input_ids=input['x_sent'], attention_mask=input['x_mask'],
                                                                  token_type_ids=input.get('token_type_ids', None),
                                                                  output_hidden_states=return_hidden_states)
        if return_hidden_states:
            seq_repre, deep_repre, hidden_states = outputs
        else:
            seq_repre, deep_repre = outputs
        deep_repre1 = self.dropout(deep_repre)
        pred_y = self.classifier(deep_repre1)

        if return_hidden_states:
            return pred_y, deep_repre, seq_repre, hidden_states
        else:
            return pred_y, deep_repre, seq_repre

    def custom_loss(self):
        return 0.0

    def forward_emb(self, input_ori):
        input, input_mask, input_pos = input_ori['x_sent'], input_ori['x_mask'], input_ori['x_pos']
        return self.pre_emb_model.embeddings.word_embeddings(input)  #self.RepresentEmb(input, input_mask, input_pos, input_typeid=input_ori['x_typeid'])

    def forward_long(self, input_embedding_all, input_ori, return_hidden_states=False):

        deep_repre, seq_repre, hidden_states = self.RepresentLongterm(input_embedding_all, input_ori=input_ori)
        deep_repre1 = self.dropout(deep_repre)
        pred_y = self.classifier(deep_repre1)

        if return_hidden_states:
            return pred_y, deep_repre, seq_repre, hidden_states
        else:
            return pred_y, deep_repre, seq_repre



from torchvision.models import resnet50
class BigModel_with_Image_Input(nn.Module):

    def __init__(self, w2id):
        super(BigModel, self).__init__(w2id)

        self.resnet=resnet50(pretrained=True)

        self.classifier=nn.Linear(768+1024,Config.class_num)

    def forward_long(self, input_embedding_all, input_ori, return_hidden_states=False):
        if Config.is_pair and not Config.use_pre_train:
            input_embedding_1, input_embedding_2 = input_embedding_all[0], input_embedding_all[1]

        if Config.is_pair and not Config.use_pre_train:
            input, input_mask, input_pos, input2, input_mask2, input_pos2 = input_ori['x_sent'], input_ori['x_mask'], \
                                                                            input_ori[
                                                                                'x_pos'], input_ori['x_sent2'], \
                                                                            input_ori['x_mask2'], input_ori['x_pos2']
        elif not Config.use_pre_train:
            input, input_mask, input_pos, = input_ori['x_sent'], input_ori['x_mask'], input_ori['x_pos']

        if Config.is_pair and not Config.use_pre_train:
            deep_repre1 = self.RepresentLongterm(input_embedding_1, input_ori=input_ori['x_sent'],
                                                 input_mask=input_ori['x_mask'], input_pos=input_ori['x_pos'])
            deep_repre2 = self.RepresentLongterm(input_embedding_2, input_ori=input_ori['x_sent2'],
                                                 input_mask=input_ori['x_mask2'], input_pos=input_ori['x_pos2'])
            deep_repre = torch.cat(
                (deep_repre1, deep_repre2, torch.abs(deep_repre1 - deep_repre2), deep_repre1 * deep_repre2), dim=-1)
        else:
            if Config.use_pre_train:
                deep_repre, seq_repre, hidden_states = self.RepresentLongterm(input_embedding_all, input_ori=input_ori)
            else:
                deep_repre, seq_repre = self.RepresentLongterm(input_embedding_all, input_ori=input_ori['x_sent'],
                                                               input_mask=input_ori['x_mask'],
                                                               input_pos=input_ori['x_pos'])
        image_repre = self.resnet(input_ori['image'])
        deep_repre1 = self.dropout(torch.cat((deep_repre,image_repre),-1))
        pred_y = self.classifier(deep_repre1)

        if return_hidden_states:
            return pred_y, deep_repre, seq_repre, hidden_states
        else:
            return pred_y, deep_repre, seq_repre