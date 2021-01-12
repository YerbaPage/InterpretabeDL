from DataBunch import Word2Id
import torch
import random
from numpy import linalg as LA
import copy
import faiss
import numpy as np
from nltk.corpus import stopwords

res = faiss.StandardGpuResources()
import logging
from Config_File import Config


def compute_aligned_mask(deep_repre, threshold, x_typeid):  # bsz, seq_len, feat_num
    align_matrix = torch.sum(deep_repre.unsqueeze(1) * deep_repre.unsqueeze(2), -1,
                             keepdim=False)  # bsz, seq_len, seq_len
    norm_matrix = torch.sum(deep_repre * deep_repre, -1, keepdim=False).pow(0.5)  # bsz, seq_len
    align_matrix /= norm_matrix.unsqueeze(1) * norm_matrix.unsqueeze(2)  # bsz, seq_len, seq_len
    # align_matrix-=torch.eye(deep_repre.size(1)).cuda().unsqueeze(0)

    typeid_mask = x_typeid.unsqueeze(1).ne(x_typeid.unsqueeze(-1)).float()
    align_matrix *= typeid_mask

    ret = align_matrix.gt(threshold)  # bsz, seq_len, seq_len
    return ret


class Analogy_Auger(Word2Id):
    def __init__(self, path, base=0, use_probase=False):
        super(Analogy_Auger, self).__init__(path, base)
        self.word_hypernym = {}
        self.word_hyponym = {}
        self.word_hypernym_p = {}
        self.word_hyponym_p = {}
        self.word_sibling = {}
        self.word_sibling_indexer = {}
        self.use_probase = use_probase

    def ReadStopConcepts(self, stop_file):
        ret = {'game': 1, 'class': 1, 'film': 1, 'product': 1, 'magazine': 1, 'activity': 1, 'book': 1, 'show': 1,
               'kind': 1, 'image': 1}
        with open(stop_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                stop_c = line.strip()
                ret[stop_c] = 1
        reader.close()
        return ret

    def ReadWordNet(self, wordnet_triple_file):
        if self.use_probase:
            self.ReadProbase(wordnet_triple_file)
            return
        sw = set(stopwords.words("english"))
        with open(wordnet_triple_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = line.replace('_', ' ')
                tokens = line.strip().split('\t')
                if tokens[0] == 'class' or tokens[2] == 'class':
                    continue
                if tokens[0] in sw or tokens[2] in sw:
                    continue

                if tokens[1] == 'hyponym':
                    if tokens[0] not in self.word_hyponym:
                        self.word_hyponym[tokens[0]] = []
                    self.word_hyponym[tokens[0]].append(tokens[2])
                if tokens[1] == 'hypernym':
                    if tokens[0] not in self.word_hypernym:
                        self.word_hypernym[tokens[0]] = []
                    self.word_hypernym[tokens[0]].append(tokens[2])
            reader.close()
        for word in self.word_hypernym:
            if word not in self.word2id:
                continue
            sibling_set = {}
            for hyperhym in self.word_hypernym[word]:
                for hyponym in self.word_hyponym[hyperhym]:
                    sibling_set[hyponym] = 1
            self.word_sibling[word] = []
            this_sibling_vec = []
            for sibling in sibling_set:
                if sibling not in self.word2id:
                    continue
                self.word_sibling[word].append(sibling)
                this_sibling_vec.append(np.expand_dims(self.vec[self.word2id[sibling]], 0))
            this_sibling_vec = np.concatenate(this_sibling_vec, axis=0)

            self.word_sibling_indexer[word] = faiss.IndexFlatL2(this_sibling_vec.shape[1])
            self.word_sibling_indexer[word].add(this_sibling_vec)

    def ReadProbase(self, probase_path):
        stop_c = self.ReadStopConcepts(Config.stop_concept_path)
        word_hypernym = {}
        word_hyponym = {}
        sw = set(stopwords.words("english"))
        with open(probase_path, 'r', encoding='utf-8') as reader:
            for line in reader:
                tokens = line.strip().split('\t')
                hypernym, hyponym, count = tokens[0], tokens[1], float(tokens[2])
                if hypernym in stop_c:
                    continue
                if (hypernym not in self.word2id) or (hyponym not in self.word2id):
                    continue
                if hypernym in sw or hyponym in sw:
                    continue
                if count <= 3.0:
                    break
                if hyponym not in word_hypernym:
                    word_hypernym[hyponym] = {}
                if hypernym not in word_hyponym:
                    word_hyponym[hypernym] = {}

                word_hypernym[hyponym][hypernym] = count
                word_hyponym[hypernym][hyponym] = count
        reader.close()
        for word in word_hyponym:
            fm = 0.0
            key_list = []
            for hyponym in word_hyponym[word]:
                fm += 1.0/word_hyponym[word][hyponym]
                key_list.append(hyponym)
            self.word_hyponym[word] = []
            self.word_hyponym_p[word] = []
            for hyponym in key_list:
                word_hyponym[word][hyponym] /= fm
                self.word_hyponym[word].append(hyponym)
                self.word_hyponym_p[word].append(1.0/word_hyponym[word][hyponym])
        for word in word_hypernym:
            fm = 0.0
            key_list = []
            for hypernym in word_hypernym[word]:
                fm += 1.0/word_hypernym[word][hypernym]
                key_list.append(hypernym)
            self.word_hypernym[word] = []
            self.word_hypernym_p[word] = []
            for hypernym in key_list:
                word_hypernym[word][hypernym] /= fm
                self.word_hypernym[word].append(hypernym)
                self.word_hypernym_p[word].append(1.0/word_hypernym[word][hypernym])

    def GenerateRandomSiblingViaProbase(self, word, hyper=None):
        if hyper not in self.word_hypernym:
            hyper = random.choices(self.word_hypernym[word], self.word_hypernym_p[word])[0]
        #return random.choices(self.word_hyponym[hyper], self.word_hyponym_p[hyper])[0], hyper
        return random.choice(self.word_hyponym[hyper]), hyper

    def aug_word_pair(self, word1, word2, origin_sentence=None):
        stop_list = ['a', 'an', 'people', 'person', 'man', 'woman', 'men', 'women']
        hyper1= '_'
        if (word1 in stop_list) or (word2 in stop_list):
            return word1, word2, '_', '_'
        if word1[0] == '[' or word2[0] == '[':
            return word1, word2, '_', '_'
        if self.use_probase:
            if (word1 not in self.word_hypernym) or (word2 not in self.word_hypernym):
                return word1, word2, '_', '_'
            word1_new, hyper1 = self.GenerateRandomSiblingViaProbase(word1)
        else:
            if (word1 not in self.word_sibling) or (word2 not in self.word_sibling):
                return word1, word2, '_', '_'
            word1_new = self.word_sibling[word1][random.randint(0, len(self.word_sibling[word1]) - 1)]
        if word1 == word2:
            return word1_new, word1_new, hyper1, hyper1
        vec1_new = self.vec[self.word2id[word1_new]] - self.vec[self.word2id[word1]] + self.vec[self.word2id[word2]]
        dist = None
        word2_new = None
        hyper2 = '-'
        '''for sibling2 in self.word_sibling[word2]:
            vec2_new = self.vec[self.word2id[sibling2]]
            this_dist = LA.norm(vec2_new - vec1_new)
            if (dist is None) or (this_dist < dist):
                dist = this_dist
                word2_new = sibling2'''
        if not self.use_probase:
            D, I = self.word_sibling_indexer[word2].search(np.expand_dims(vec1_new, 0), 1)
            word2_new = self.word_sibling[word2][I[0][0]]
            if origin_sentence is not None:
                index = origin_sentence.index('[SEP]')
                sent1 = ' '.join(origin_sentence[:index])
                sent2 = ' '.join(origin_sentence[index + 1:])
        else:
            for i in range(20):
                sibling2, hyper2 = self.GenerateRandomSiblingViaProbase(word2, hyper1)
                vec2_new = self.vec[self.word2id[sibling2]]
                this_dist = LA.norm(vec2_new - vec1_new)
                if (dist is None) or (this_dist < dist):
                    dist = this_dist
                    word2_new = sibling2
        return word1_new, word2_new, hyper1, hyper2

    def augment(self, batch_data, net, tokenizer, deep_repre=None):
        if deep_repre is None:
            _, deep_repre = net(batch_data, return_repres=True)
        aligned_mask = compute_aligned_mask(deep_repre, Config.aug_threshold, batch_data['x_typeid'])
        bsz = aligned_mask.size(0)

        batch_data_aug = copy.deepcopy(batch_data)
        for i in range(bsz):
            if torch.sum(aligned_mask[i]) > 0:
                for j1 in range(aligned_mask.size(1)):
                    j2 = torch.argmax(aligned_mask[i][j1], -1).item()
                    if aligned_mask[i][j1][j2] > 0:
                        word1 = tokenizer.convert_ids_to_tokens([batch_data['x_sent'][i][j1]])
                        word2 = tokenizer.convert_ids_to_tokens([batch_data['x_sent'][i][j2]])
                        ori_sent = tokenizer.convert_ids_to_tokens(batch_data['x_sent'][i])
                        word1_new, word2_new, hyper1, hyper2 = self.aug_word_pair(word1[0], word2[0], ori_sent)
                        if word2_new != word2[0] or word1_new != word1[0]:
                            logging.warning(
                                "{}\nprev:{},{}\tnew:{},{}\thyper:{},{}\n".format(
                                    ' '.join(ori_sent).replace(' [PAD]', ''), word1,
                                    word2, word1_new, word2_new, hyper1, hyper2))
                        id1_new = tokenizer.convert_tokens_to_ids(word1_new)
                        id2_new = tokenizer.convert_tokens_to_ids(word2_new)
                        #id1_new = tokenizer.convert_tokens_to_ids(hyper1)
                        #id2_new = tokenizer.convert_tokens_to_ids(hyper2)
                        #id1_new = 10
                        #id2_new = 10
                        batch_data_aug['x_sent'][i][j1] = id1_new
                        batch_data_aug['x_sent'][i][j2] = id2_new
        return batch_data_aug
