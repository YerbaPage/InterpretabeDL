from torch.utils.data import Dataset
import torch
import numpy as np
import ljqpy
from Config_File import Config
from nltk.tokenize import word_tokenize
import os
import h5py
import time
from random import choice as randchoice
from random import randint, random
from torch.utils import data
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from modules import ComputeRelative

from DataBunch import DataBunch

import os
from skimage import io, transform
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import pickle

class DataBunch_with_image_for_joint_embedding(DataBunch):
    def __init__(self, inputFile, w2id_c, sentence_token, label_token, sentence2_token=None, case_sensitive=False,
                 dataset='SNLI_visual_representation_learning', need_tokenize=True, image_folder=Config.image_folder, id_token=None):
        self.index_imageindex_file = inputFile + '_' + self.__class__.__name__ + 'index_imageindex.pkl'
        super(DataBunch_with_image_for_joint_embedding, self).__init__(inputFile, w2id_c, sentence_token, label_token,
                                                                       sentence2_token, case_sensitive,
                                                                       dataset, need_tokenize, id_token=id_token)
        if self.loaded:
            return


        self.index_imageindex={}
        id_index = {}
        with open(inputFile, 'r', encoding='utf-8') as reader:
            last_index = 0
            for index, line in enumerate(reader):
                if index == 0:
                    continue
                if Config.dataset=='mini-SNLI':
                    tokens = line.strip().split('#')
                    if len(tokens) <= 1:
                        continue
                    id = tokens[1][2:-4]
                else:
                    tokens = line.strip().split('\t')
                    if len(tokens) <= 1:
                        continue
                    id = tokens[1][:-6]
                if id not in id_index:
                    id_index[id] = []
                id_index[id].append(index - 1)
                last_index = index - 1
            # print(last_index)
            # print(len(self.data['x_sent']))
            reader.close()
        image_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip()])
        #self.data['image'] = [None] * len(self.data['x_sent'])
        self.data['image_list'] = []
        for file in os.listdir(image_folder):
            if file.endswith(".jpg"):
                id = file[:-4]
                if id not in id_index:
                    continue
                file_path = os.path.join(image_folder, file)
                image = Image.open(file_path).convert('RGB')
                sample = image_transform(image)
                sample = np.array(sample)
                self.data['image_list'].append(sample)
                for index in id_index[id]:
                    self.index_imageindex[index]=len(self.data['image_list'])-1
                    #self.data['image'][index] = sample
        #image_size = len(self.data['image'])
        '''zero_count=0
        for i in range(image_size):
            if self.data['image'][i] is None:
                self.data['image'][i]=np.zeros((64,64,3))
                zero_count+=1
        print('zero:total image {}:{}'.format(zero_count,len(self.data['image'])))'''

        for key in self.data:
            self.data[key] = np.array(self.data[key])
        #self.data['image'] = np.transpose(self.data['image'], [0, 3, 1, 2])
        self.data['image_list'] = np.transpose(self.data['image_list'], [0, 3, 1, 2])
        self.Save()

    def __getitem__(self, index):
        ret = {}
        for key in self.data:
            if key == 'image_list':
                if index in self.index_imageindex:
                    ret['image'] = torch.FloatTensor(self.data[key][self.index_imageindex[index]])  # .to(Config.device)
                else:
                    ret['image'] = torch.zeros((3,64,64))
                continue
            if key!='image_list' and len(self.data[key]) <= index:
                continue
            elif key == 'Id':
                ret[key] = str(np.array(self.data['Id'][index]).astype(str))
            elif key.startswith('txt'):
                ret[key] = self.data[key][index]
            else:
                ret[key] = torch.LongTensor(self.data[key][index])  # .to(Config.device)

        if '_lm' in self.dataset:
            ret['y'] = torch.LongTensor(self.data['x_sent'][:, index + 1])  # .to(Config.device)

        return ret

    def Save(self):
        string_dt = h5py.special_dtype(vlen=str)
        with h5py.File(self.h5File, 'w') as dfile:
            for key in self.data:
                if key.startswith('Id') or key.startswith('txt'):
                    dfile.create_dataset(key, data=self.data[key], dtype=string_dt)
                else:
                    dfile.create_dataset(key, data=self.data[key])

        with open(self.index_imageindex_file, 'wb') as handle:
            pickle.dump(self.index_imageindex, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    def Load(self):
        with h5py.File(self.h5File) as dfile:
            for key in dfile:
                self.data[key] = dfile[key][:]
                print('Loaded {} with size {}'.format(key, str(self.data[key].shape)))
        with open(self.index_imageindex_file, 'rb') as handle:
            self.index_imageindex=pickle.load(handle)
            handle.close()

        print('Loaded h5 from {}'.format(self.h5File))