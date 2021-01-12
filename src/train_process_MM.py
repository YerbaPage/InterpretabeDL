from Config_File import Config
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import time
import torch
from tqdm import tqdm
from DataAnalysis import ViewTopKNearest
from utils import get_acc, TestBIO, transpose_batch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from model import MLP
import torch.nn as nn
import torch.nn.functional as F
import math
import optimizers
from train_process import evaluate
from network import BigModel
from Config_File import Config
from model import Classifier

class BigModel_two_bert(BigModel):
    def __init__(self, w2id):
        super(BigModel_two_bert, self).__init__(w2id)
        self.pre_emb_model2 = Config.pre_trained_model[0].from_pretrained(Config.pre_trained_model[2],
                                                                         config=Config.pre_train_config)
        if Config.deep_model != '':
            if Config.deep_model.endswith('CRF'):
                classifier_dim = self.deep_model[-2].d_model
            else:
                classifier_dim = self.deep_model[-1].d_model
        else:
            classifier_dim = Config.word_dim
        self.concat_classifier = Classifier(classifier_dim * 2, Config.class_num, dropout=0)

    def forward(self, input, return_hidden_states=False):
        temp = self.forward_emb(input)
        #return self.forward_long(temp, input, return_hidden_states)
        pred_y, deep_repre, seq_repre, *_ = self.forward_long(temp, input, return_hidden_states)
        #return pred_y
        _,deep_repre2, *_=self.pre_emb_model2(input['x_sent'],input['x_mask'],input['x_typeid'],input['x_pos'])

        deep_repre2 = torch.cat((deep_repre,deep_repre2),-1)
        pred_y2 = self.concat_classifier(deep_repre2)
        return pred_y2, deep_repre, seq_repre


def train_MM_from_LXMERT(model, optimizer, scheduler, criterion, train_generator, test_generator):
    best_accu = 0
    best_f1 = 0
    best_ppl = 1000000
    accu_min_train_loss = 0
    last_update_epoch = 0
    batch_count = 0

    state_dict = torch.load(Config.pre_trained_model_path)
    if isinstance(model, torch.nn.DataParallel):
        model.module.pre_emb_model.load_state_dict(state_dict)
    else:
        model.pre_emb_model.load_state_dict(state_dict)

    # optimizer_rho=optimizers.__dict__[Config.optimizer](model.perturb.bias, int(
    #            len(trainset[Config.dataset_train]) / params['batch_size']) * Config.epoch,lr=5e-3)

    for epoch in range(Config.epoch):
        if (Config.early_stop is not None) and (epoch - last_update_epoch > Config.early_stop):
            break
        train_loss = 0
        count = 0
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        priores = AverageMeter()
        varies = AverageMeter()
        bar = Bar('Processing', max=len(train_generator))
        end = time.time()
        with torch.autograd.set_detect_anomaly(True):
            # for batch_data  in tqdm(train_generator,total=len(train_generator),desc='train'):
            for batch_data0 in train_generator:
                if '_lm' not in Config.dataset:
                    batch_data = {}
                    for k in batch_data0:
                        v = batch_data0[k]
                        if isinstance(v, torch.LongTensor) or isinstance(v, torch.FloatTensor):
                            batch_data[k] = v.cuda()
                        else:
                            batch_data[k] = v
                else:
                    batch_data = transpose_batch(batch_data0)

                data_time.update(time.time() - end)
                batch_count += 1
                count += 1
                # if count+1==83:
                #    print('cwy debug')
                model.train()
                optimizer.zero_grad()
                # optimizer_rho.zero_grad()
                model.zero_grad()

                local_labels = batch_data['y'].to(Config.device).squeeze()

                pred_y, *_ = model(batch_data)
                loss = criterion(pred_y.reshape(-1, pred_y.shape[-1]), local_labels.reshape(-1))

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), Config.clip)
                train_loss += loss
                optimizer.step()
                # optimizer_rho.step()
                scheduler.step()

                losses.update(loss.data, batch_data['x_sent'].size(0))
                if count % 20 == 1:
                    top1.update(get_acc(pred_y, local_labels), batch_data['y'].size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {accu: .4f}'.format(
                    batch=count + 1, size=len(train_generator), data=data_time.avg, bt=batch_time.avg,
                    total=bar.elapsed_td, eta=bar.eta_td, loss=losses.avg, accu=top1.avg)
                bar.next()
            bar.finish()

        print("epoch:{} train_loss:{}".format(epoch, train_loss / len(train_generator)))

        accuracy, f1, val_loss = evaluate(model, criterion, test_generator)
        if '_lm' in Config.dataset:
            ppl = math.exp(val_loss)
            if ppl < best_ppl:
                best_accu = accuracy
                last_update_epoch = epoch
                torch.save(model.state_dict(), Config.model_save_path)
                best_ppl = ppl
                print('update new model at {}'.format(Config.model_save_path))
        else:
            if best_accu < accuracy:
                best_accu = accuracy
                last_update_epoch = epoch
                best_f1 = f1
                torch.save(model.state_dict(), Config.model_save_path)
                print('update new model at {}'.format(Config.model_save_path))

        resultStr = "mode:{} epoch:{} val_loss:{:.4f} accu:{:.4f}, f1:{}, best accu:{:.4f}, minaccu:{:.4f}".format(
            Config.config_name, epoch, val_loss, accuracy, f1, best_accu, accu_min_train_loss)
        if '_lm' in Config.dataset:
            resultStr += ' ppl:{:.4f} min ppl:{:.4f}'.format(ppl, best_ppl)
        print(resultStr)
    if Config.class_num == 2:
        return best_accu, best_f1
    else:
        return best_accu

