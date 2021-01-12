from torch.utils.data import Dataset
import torch
import numpy as np
import ljqpy
from nltk.tokenize import word_tokenize
import os
import h5py
import time
from random import choice as randchoice
from random import randint, random
from torch.utils import data
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from modules import ComputeRelative


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.shape[0] // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[:nbatch * bsz]  # .narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.reshape((bsz, -1))
    return data


class Word2Id():
    def __init__(self, path, base=0):
        self.base = base
        self.word2id = {}
        self.id2word = {}
        self.vec = []
        self.vec_path = path
        self.word_range = {}

    def SaveWordRange(self, outputFile):
        with open(outputFile, 'w', encoding='utf-8') as writer:
            for word in self.word_range:
                writer.write('{}\n'.format(word))
            writer.close()

    def ReadWordRange(self, inputFile):
        with open(inputFile, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = line.strip()
                self.word_range[line] = 1
            reader.close()

    def ReadAllWords(self, inputFile, indexes, case_sensitive=False, need_tokenize=False):
        line_count = 0
        for line in ljqpy.LoadCSVg(inputFile):
            line_count += 1
            if line_count % Config.few_count == 0 and Config.load_few:
                break

            for index in indexes:
                if index == None:
                    continue
                if index >= len(line):
                    continue
                if not case_sensitive:
                    line[index] = line[index].lower()
                if not need_tokenize:
                    words = line[index].split(' ')
                else:
                    words = word_tokenize(line[index])
                for word in words:
                    self.word_range[word] = 1

    def ReadPretrainedDict(self):
        f = open(self.vec_path, 'r', encoding="utf-8")
        self.word2id['<PAD>'] = 0
        self.vec.append([random() for i in range(Config.word2vec_dim)])
        self.word2id['<CLS>'] = len(self.word2id)
        self.word2id['<SEP>'] = len(self.word2id)
        self.word2id['<UNK>'] = len(self.word2id)
        self.vec.append([random() for i in range(Config.word2vec_dim)])
        self.vec.append([random() for i in range(Config.word2vec_dim)])
        self.vec.append([random() for i in range(Config.word2vec_dim)])

        for i in range(Config.sent_len):
            self.word2id['<POS>' + str(i)] = len(self.word2id)
            self.vec.append([random() for i in range(len(self.vec[0]))])
        while True:
            line = f.readline()
            if line == '':
                break
            content = line.strip().split(' ')
            if len(self.word_range) > 0 and (content[0] not in self.word_range):
                continue
            self.word2id[content[0]] = len(self.word2id)
            content = content[1:]
            if len(content) != Config.word2vec_dim:
                continue
            content = [(float)(i) for i in content]
            self.vec.append(content)
            # break
        self.vec = np.array(self.vec).astype('float32')
        print(self.vec.shape)
        f.close()

        for word in self.word2id:
            id = self.word2id[word]
            self.id2word[id] = word


class DataBunch(Dataset):
    def ParseLabel(self, args, label):
        if self.dataset in ['RTE', 'QNLI']:
            if label == 'not_entailment':
                return [0]
            else:
                return [1]
        if self.dataset in ['PTB', 'NER', 'SRL']:
            tags = label.split(' ')
            ret = []
            for tag_str in tags:
                tag = int(tag_str)
                ret.append([tag])
                if len(ret) == args.sent_len:
                    break
            while len(ret) < args.sent_len:
                ret.append([args.class_num_dict[self.dataset] - 1])
            return ret
        if self.dataset in ['SST', 'CoLA', 'QQP', 'SST5', 'mtl-baby', 'SST5-aug-finetuned-GPT2']:
            return [int(label)]
        if self.dataset == 'MSRP':
            return [int(label)]
        if self.dataset in ['MNLI', 'SNLI', 'mini-SNLI', 'e-snli']:
            if label == 'neutral':
                return [0]
            if label == 'entailment':
                return [1]
            if label == 'contradiction':
                return [2]

    def ParseSentence(self, args, tokens, tokenizer=None):
        output_sent = []
        output_mask = []
        ret = {}
        if args.is_pair:
            sentence1=tokens[args.sent_token_dict[args.dataset]]
            sentence2 = tokens[args.sent2_token_dict[args.dataset]]
            inputs = tokenizer.encode_plus(sentence1, sentence2, add_special_tokens=True,max_length=args.sent_len,
                                           padding='max_length', truncation=True)
        else:
            sentence1=tokens[args.sent_token_dict[args.dataset]]
            inputs = tokenizer.encode_plus(sentence1, add_special_tokens=True, max_length=args.sent_len,
                                           padding='max_length',truncation=True)
        output_sent, output_mask = inputs['input_ids'], inputs['attention_mask']
        if 'token_type_ids' in inputs:
            output_typeid = inputs['token_type_ids']
        else:
            output_typeid = None
        ret['x_sent'] = output_sent
        ret['x_mask'] = output_mask
        if output_typeid is not None:
            ret['x_typeid'] = output_typeid
        if args.is_pair:
            ret['txt'] = (sentence1+'[SEP]'+sentence2).encode("ascii", "ignore")
        else:
            ret['txt'] = sentence1.encode("ascii", "ignore")
        return ret


    def __init__(self, args, inputFile, sentence_token, label_token, tokenizer, sentence2_token=None,
                 dataset='RTE', id_token=None, load_few=False, **kwargs):
        print('start reading file{}'.format(inputFile))
        self.loaded = False
        self.h5File = inputFile + '_Pretrain_' + self.__class__.__name__ + args.model_name_or_path
        if args.load_few:
            self.h5File += '_loadfew'
        self.h5File += '.h5'

        self.is_pair = (sentence2_token is not None)
        self.label_id = {}
        self.dataset = dataset
        self.data = {'y': [], 'Id': []}

        if os.path.exists(self.h5File) and args.load_cached_h5:
            self.Load()
            self.loaded = True

            print('{} loaded from h5'.format(inputFile))
            return

        line_count = 0
        for tokens in ljqpy.LoadCSVg(inputFile):
            line_count += 1
            if line_count == 1:
                continue
            if line_count % 1000 == 0:
                print('loading line {} for file {}'.format(line_count, inputFile))
            if load_few and line_count == args.few_count:
                break
            if len(tokens)<=1:
                continue

            if id_token is not None:
                self.data['Id'].append(tokens[id_token])

            output = self.ParseSentence(args,tokens,tokenizer)
            for key in output:
                if key not in self.data:
                    self.data[key] = []
                self.data[key].append(output[key])

            if label_token is not None:
                label = self.ParseLabel(args,tokens[label_token])
                self.data['y'].append(label)
            else:
                self.data['y'].append([0])

        keys = [key for key in self.data]
        for key in keys:
            if key == 'Id' or key.startswith('txt'):
                self.data[key] = np.array(self.data[key], dtype=object)
            else:
                self.data[key] = np.array(self.data[key], dtype=int)

        self.Save()

        print('{} loaded from raw data'.format(inputFile))

    def __len__(self):
        return len(self.data['x_sent'])

    def __getitem__(self, index):
        ret = {}
        for key in self.data:
            if len(self.data[key]) <= index:
                continue
            if key == 'Id':
                ret[key] = str(np.array(self.data['Id'][index]).astype(str))
            elif key.startswith('txt'):
                ret[key] = self.data[key][index]
            else:
                ret[key] = torch.LongTensor(self.data[key][index])  # .to(Config.device)

        return ret

    def Save(self):
        string_dt = h5py.special_dtype(vlen=str)
        with h5py.File(self.h5File, 'w') as dfile:
            for key in self.data:
                if key.startswith('Id') or key.startswith('txt'):
                    dfile.create_dataset(key, data=self.data[key], dtype=string_dt)
                else:
                    dfile.create_dataset(key, data=self.data[key])

    def Load(self):
        with h5py.File(self.h5File) as dfile:
            for key in dfile:
                self.data[key] = dfile[key][:]
                print('Loaded {} with size {}'.format(key, str(self.data[key].shape)))
        print('Loaded h5 from {}'.format(self.h5File))

class DataBunch_e_snli(DataBunch):
    def ParseSentence(self, args, tokens, tokenizer=None):
        output_sent = []
        output_pos = []
        output_mask = []
        ret = {}
        #sent1_token,sent2_token,exp_token=[int(t) for t in args.sent_token_dict['e-snli'].split(';')]
        #sentence1, sentence2, explanation = tokens[sent1_token],tokens[sent2_token],tokens[exp_token]
        #inputs = tokenizer.encode_plus(sentence1, sentence2+explanation, add_special_tokens=True,
        sentence1 = tokens[args.sent_token_dict[args.dataset]]
        sentence2 = tokens[args.sent2_token_dict[args.dataset]]
        explanation = tokens[4]
        inputs = tokenizer.encode_plus(sentence1, sentence2+explanation, add_special_tokens=True,
                                                                max_length=args.sent_len, pad_to_max_length=True,truncation=True)
        output_sent, output_mask = inputs['input_ids'], inputs['attention_mask']
        if 'token_type_ids' in inputs:
            output_typeid = inputs['token_type_ids']
        else:
            output_typeid = None
        ret['x_sent'] = output_sent
        ret['x_mask'] = output_mask
        if output_typeid is not None:
            ret['x_typeid'] = output_typeid
        if args.is_pair:
            ret['txt'] = (sentence1 + '[SEP]' + sentence2).encode("ascii", "ignore")
        else:
            ret['txt'] = sentence1.encode("ascii", "ignore")
        return ret

class DataBunch_e_snli_marked(DataBunch_e_snli):
    def ParseSentence(self, args, tokens, tokenizer=None):
        output_sent = []
        output_pos = []
        output_mask = []
        ret = {}
        sentence1 = tokens[args.sent_token_dict[args.dataset]]
        sentence2 = tokens[args.sent2_token_dict[args.dataset]]
        inputs = tokenizer.encode_plus(sentence1, sentence2 , add_special_tokens=True,
                                       max_length=args.sent_len, pad_to_max_length=True, truncation=True)
        output_sent, output_mask = inputs['input_ids'], inputs['attention_mask']
        if 'token_type_ids' in inputs:
            output_typeid = inputs['token_type_ids']
        else:
            output_typeid = None

        for i in range(args.sent_len):
            output_pos.append(i + 1 if output_mask[i] == 1 else 0)
        ret['x_sent'] = output_sent
        ret['x_pos'] = output_pos
        ret['x_mask'] = output_mask
        if output_typeid is not None:
            ret['x_typeid'] = output_typeid
        if args.is_pair:
            ret['txt'] = (sentence1 + '[SEP]' + sentence2).encode("ascii", "ignore")
        else:
            ret['txt'] = sentence1.encode("ascii", "ignore")

        cause_index1={}
        cause_index2 = {}
        cause_words1, cause_words2={}, {}
        if len(tokens)==10:
            cause_tokens1 = [6]
            cause_tokens2 = [7]
        else:
            cause_tokens1=[5]
            cause_tokens2 = [6]
        ret['cause_mask']=np.zeros(len(ret['x_sent']),)

        for index in cause_tokens1:
            sent=tokens[index].replace('  ',' ')
            words=sent.strip().split(' ')
            for j,word in enumerate(words):
                if word == '':
                    continue
                if word[0]=='*' and word[-1]=='*':
                    t_word=word[1:-1].strip()
                    if t_word[-1]=='.':
                        t_word=t_word[:-1]
                    cause_index1[j]=1
                    cause_words1[t_word]=1

        for index in cause_tokens2:
            sent=tokens[index].replace('  ',' ')
            words=sent.strip().split(' ')
            for j,word in enumerate(words):
                if word=='':
                    continue
                if word[0]=='*' and word[-1]=='*':
                    t_word = word[1:-1].strip()
                    if t_word[-1] == '.':
                        t_word = t_word[:-1]
                    cause_index2[j] = 1
                    cause_words2[t_word] = 1

        is_first=True
        j=-1
        for i in range(len(ret['x_sent'])):
            this_word=tokenizer.convert_ids_to_tokens(ret['x_sent'][i]).strip()

            if this_word.startswith('##'):
                l=i-1
                while l>=0 and tokenizer.convert_ids_to_tokens(ret['x_sent'][l+1]).strip().startswith('##'):
                    this_word=tokenizer.convert_ids_to_tokens(ret['x_sent'][l])+this_word
                    l-=1
            elif ret['x_sent'][i]!=tokenizer.cls_token_id:
                j+=1
            l = i + 1
            while l < len(ret['x_sent']) and tokenizer.convert_ids_to_tokens(ret['x_sent'][l]).strip().startswith('##'):
                this_word = this_word + tokenizer.convert_ids_to_tokens(ret['x_sent'][l])
                l += 1
            this_word = this_word.replace('#','')

            if is_first:
                if j in cause_index1 and this_word in cause_words1:
                    ret['cause_mask'][i]=1
                else:
                    ret['cause_mask'][i]=0
            else:
                if j in cause_index2 and this_word in cause_words2:
                    ret['cause_mask'][i]=1
                else:
                    ret['cause_mask'][i]=0
            if ret['x_sent'][i]==tokenizer.sep_token_id:
                is_first=False
                j=-1
        return ret