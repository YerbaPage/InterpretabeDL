from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import time
import torch
from tqdm import tqdm
from utils import get_acc, TestBIO, transpose_batch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from model import MLP
import torch.nn as nn
import torch.nn.functional as F
import math
import optimizers
from train_process_causality import train_cause_word

def evaluate(args, model, criterion, test_generator, returns_metrics):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    bar = Bar('Testing', max=len(test_generator))
    end = time.time()
    val_loss = 0
    pred_y_all = []
    labels_all = []
    model.eval()
    count = 0

    total_count = 0
    correct_count = 0
    pred_count = 0

    with torch.no_grad():
        for batch_data0 in test_generator:
            # batch_data = batch_data0
            batch_data = {}
            for k in batch_data0:
                v = batch_data0[k]
                if isinstance(v, torch.LongTensor) or isinstance(v, torch.FloatTensor):
                    batch_data[k] = v.cuda()
                else:
                    batch_data[k] = v

            count += 1

            data_time.update(time.time() - end)
            model.zero_grad()

            local_labels = batch_data['y'].squeeze()

            pred_y, *_ = model(batch_data)

            if args.dataset in ['NER', 'SRL']:
                t1, t2, t3 = TestBIO(pred_y, local_labels)
                total_count += t1
                pred_count += t2
                correct_count += t3

            pred_y = pred_y.reshape(-1, pred_y.shape[-1])
            local_labels = local_labels.reshape(-1)
            val_loss += criterion(pred_y, local_labels)

            _, pred_y_label = torch.max(pred_y, 1)
            pred_y_label = pred_y_label.detach().cpu().numpy()

            for i, t in enumerate(pred_y_label):
                if args.is_sequence:
                    if local_labels[i] != args.pos_tag_count - 1:
                        pred_y_all.append(t)
                        labels_all.append(local_labels[i].item())
                else:
                    pred_y_all.append(t.item())
                    labels_all.append(local_labels[i].item())

            losses.update(criterion(pred_y, local_labels).data, batch_data['y'].size(0))
            if count % 20 == 1:
                top1.update(get_acc(args, pred_y, local_labels), batch_data['y'].size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {accu: .4f}'.format(
                batch=count + 1, size=len(test_generator), data=data_time.avg, bt=batch_time.avg, total=bar.elapsed_td,
                eta=bar.eta_td, loss=losses.avg, accu=top1.avg)
            bar.next()
        bar.finish()

        labels_all = np.array(labels_all)
        pred_y_all = np.array(pred_y_all)
        accuracy = accuracy_score(labels_all, pred_y_all)
        if args.class_num == 2:
            f1 = f1_score(labels_all, pred_y_all)
        else:
            f1 = None
        if args.dataset in ['NER', 'SRL'] and total_count > 0 and pred_count > 0 and correct_count > 0:
            precision = correct_count * 1.0 / pred_count
            recall = correct_count * 1.0 / total_count
            f1 = 2 * precision * recall / (precision + recall)
            print('NER f1:{}  precision:{}  recall:{}'.format(f1, precision, recall))

        return accuracy, f1, val_loss / len(test_generator)


def train(args, model, optimizer, scheduler, criterion, train_generator, test_generator):
    best_accu = 0
    best_f1 = 0
    accu_min_train_loss = 0
    last_update_epoch = 0
    batch_count = 0

    for epoch in range(args.epoch):
        if (args.early_stop is not None) and (epoch - last_update_epoch > args.early_stop):
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
            for batch_data0 in train_generator:
                batch_data = {}
                for k in batch_data0:
                    v = batch_data0[k]
                    if isinstance(v, torch.LongTensor) or isinstance(v, torch.FloatTensor):
                        batch_data[k] = v.cuda()
                    else:
                        batch_data[k] = v

                data_time.update(time.time() - end)
                batch_count += 1
                count += 1
                model.train()
                optimizer.zero_grad()
                model.zero_grad()

                local_labels = batch_data['y'].to(args.device).squeeze()

                pred_y, *_ = model(batch_data)
                loss = criterion(pred_y.reshape(-1, pred_y.shape[-1]), local_labels.reshape(-1))

                loss.backward()
                # print('loss value',str([t for t in model.parameters()][0].grad.sum()))
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                train_loss += loss.item()
                optimizer.step()
                scheduler.step()

                losses.update(loss.data, batch_data['x_sent'].size(0))
                if count % 20 == 1:
                    top1.update(get_acc(args, pred_y, local_labels), batch_data['y'].size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {accu: .4f}'.format(
                    batch=count + 1, size=len(train_generator), data=data_time.avg, bt=batch_time.avg,
                    total=bar.elapsed_td, eta=bar.eta_td, loss=losses.avg, accu=top1.avg)
                bar.next()

            bar.finish()

        print("epoch:{} train_loss:{}".format(epoch, train_loss / len(train_generator)))

        accuracy, f1, val_loss, *_ = evaluate(args, model, criterion, test_generator, True)
        if best_accu < accuracy:
            best_accu = accuracy
            last_update_epoch = epoch
            best_f1 = f1
            torch.save(model.state_dict(), args.model_save_path)
            print('update new model at {}'.format(args.model_save_path))

        resultStr = "mode:{} epoch:{} val_loss:{:.4f} accu:{:.4f}, f1:{}, best accu:{:.4f}, minaccu:{:.4f}".format(
            args.config_name, epoch, val_loss, accuracy, f1, best_accu, accu_min_train_loss)
        print(resultStr)
    if args.class_num == 2:
        return best_accu, best_f1
    else:
        return best_accu





def train_pgd(model, optimizer, scheduler, criterion, train_generator, test_generator):
    pgd = PGD_from_zhengguangyu(model)
    best_accu = 0
    accu_min_train_loss = 0
    last_update_epoch = 0
    for epoch in range(Config.epoch):
        if (Config.early_stop is not None) and (epoch - last_update_epoch > Config.early_stop):
            break

        train_loss = 0
        count = 0
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        bar = Bar('Processing', max=len(train_generator))
        end = time.time()
        with torch.autograd.set_detect_anomaly(True):
            for batch_data0 in tqdm(train_generator, total=len(train_generator), desc='train_pgd'):
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
                local_labels = batch_data['y'].to(Config.device).squeeze()

                data_time.update(time.time() - end)
                count += 1
                model.train()
                optimizer.zero_grad()
                model.zero_grad()

                pred_y, *_ = model(batch_data)
                loss = criterion(pred_y.reshape(-1, pred_y.shape[-1]), local_labels.reshape(-1))
                loss.backward()
                pgd.backup_grad()

                if isinstance(model, torch.nn.DataParallel):
                    batch_embedding = model.module.forward_emb(batch_data)
                else:
                    batch_embedding = model.forward_emb(batch_data)

                if Config.is_pair and not Config.use_pre_train:
                    delta0 = torch.zeros_like(batch_embedding[0]).cuda()
                    delta0.requires_grad_()
                    delta1 = torch.zeros_like(batch_embedding[1]).cuda()
                    delta1.requires_grad_()
                else:
                    delta0 = torch.zeros_like(batch_embedding).cuda().uniform_(-1, 1).cuda() * batch_data[
                        'x_mask'].float().unsqueeze(-1)
                    delta0.requires_grad_()

                loss_adv = 0
                for t in range(Config.adversarial_K):
                    # compute grad for delta and parameters of the original model
                    if t == Config.adversarial_K - 1:
                        pgd.restore_grad()  # restore to the grad after the original samples forward
                    else:
                        model.zero_grad()  # clear the grads of the model, delta0's grad is already zero in its initialization

                    if Config.is_pair and not Config.use_pre_train:
                        pred_adv, *_ = model.forward_long((batch_embedding + delta0, batch_embedding[1] + delta1),
                                                          input_ori=batch_data)
                    else:
                        if isinstance(model, torch.nn.DataParallel):
                            pred_adv, *_ = model.module.forward_long(batch_embedding + delta0, input_ori=batch_data)
                        else:
                            pred_adv, *_ = model.forward_long(batch_embedding + delta0, input_ori=batch_data)
                    loss_adv = criterion(pred_adv.reshape(-1, pred_adv.shape[-1]), local_labels.reshape(-1))
                    loss_adv.backward()  # add grads of delta and parameters for this step

                    if t == Config.adversarial_K - 1:  # no need to update delta
                        break

                    # update delta according to its grad and epsilon
                    if Config.is_pair and not Config.use_pre_train:
                        delta0 = pgd.attack_on_emb(batch_embedding[0] + delta0, batch_embedding[0],
                                                   delta0.grad, is_first_attack=(t == 0)) - batch_embedding[0]
                        delta1 = pgd.attack_on_emb(batch_embedding[1] + delta1, batch_embedding[1],
                                                   delta1.grad, is_first_attack=(t == 0)) - batch_embedding[1]
                        delta0 = delta0.clone().detach()
                        delta0.requires_grad_()
                        delta1 = delta1.clone().detach()
                        delta1.requires_grad_()
                    else:
                        delta0 = pgd.attack_on_emb(batch_embedding + delta0, batch_embedding,
                                                   delta0.grad, is_first_attack=(t == 0)) - batch_embedding
                        delta0 = delta0.clone().detach()
                        delta0.requires_grad_()
                    if isinstance(model, torch.nn.DataParallel):
                        batch_embedding = model.module.forward_emb(batch_data)
                    else:
                        batch_embedding = model.forward_emb(batch_data)

                # grads of the original model has been added
                train_loss += loss_adv.item()

                optimizer.step()
                scheduler.step()
                model.zero_grad()

                losses.update(loss_adv.data, batch_data['y'].size(0))
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

        accuracy, f1, val_loss, *_ = evaluate(model, criterion, test_generator, returns_metrics=True)
        if best_accu < accuracy:
            best_accu = accuracy
            last_update_epoch = epoch
            torch.save(model.state_dict(), Config.model_save_path)

        resultStr = "mode:{} epoch:{} val_loss:{:.4f} accu:{:.4f}, f1:{}, best accu:{:.4f}, minaccu:{:.4f}".format(
            Config.config_name, epoch, val_loss, accuracy, f1, best_accu, accu_min_train_loss)
        print(resultStr)
    return best_accu


def compute_aligned_mask(deep_repre, threshold, x_typeid):  # bsz, seq_len, feat_num
    align_matrix = torch.sum(deep_repre.unsqueeze(1) * deep_repre.unsqueeze(2), -1,
                             keepdim=False)  # bsz, seq_len, seq_len
    norm_matrix = torch.sum(deep_repre * deep_repre, -1, keepdim=False).pow(0.5)  # bsz, seq_len
    align_matrix /= norm_matrix.unsqueeze(1) * norm_matrix.unsqueeze(2)  # bsz, seq_len, seq_len
    # align_matrix-=torch.eye(deep_repre.size(1)).cuda().unsqueeze(0)

    typeid_mask = x_typeid.unsqueeze(1).ne(x_typeid.unsqueeze(-1)).float()
    align_matrix *= typeid_mask

    ret = torch.sum(align_matrix.gt(threshold), -1, keepdim=True).ge(1).float()  # bsz, seq_len, 1
    return ret


def train_pgd_aligned_mask(model, optimizer, scheduler, criterion, train_generator, test_generator):
    pgd = PGD_from_zhengguangyu(model)
    best_accu = 0
    accu_min_train_loss = 0
    last_update_epoch = 0
    for epoch in range(Config.epoch):
        if (Config.early_stop is not None) and (epoch - last_update_epoch > Config.early_stop):
            break

        train_loss = 0
        count = 0
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        bar = Bar('Processing', max=len(train_generator))
        end = time.time()
        with torch.autograd.set_detect_anomaly(True):
            for batch_data0 in tqdm(train_generator, total=len(train_generator), desc='train_pgd'):
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
                local_labels = batch_data['y'].to(Config.device).squeeze()

                data_time.update(time.time() - end)
                count += 1
                model.train()
                optimizer.zero_grad()
                model.zero_grad()

                pred_y, deep_repres, seq_repre = model(batch_data)
                loss = criterion(pred_y.reshape(-1, pred_y.shape[-1]), local_labels.reshape(-1))
                loss.backward()
                pgd.backup_grad()

                aligned_mask = compute_aligned_mask(seq_repre.clone().detach(), Config.aligned_threshold,
                                                    batch_data['x_typeid'])

                if isinstance(model, torch.nn.DataParallel):
                    batch_embedding = model.module.forward_emb(batch_data)
                else:
                    batch_embedding = model.forward_emb(batch_data)

                if Config.is_pair and not Config.use_pre_train:
                    delta0 = torch.zeros_like(batch_embedding[0]).cuda()
                    delta0.requires_grad_()
                    delta1 = torch.zeros_like(batch_embedding[1]).cuda()
                    delta1.requires_grad_()
                else:
                    delta0 = torch.zeros_like(batch_embedding).cuda().uniform_(-1, 1).cuda() * batch_data[
                        'x_mask'].unsqueeze(-1).float()
                    delta0 *= aligned_mask
                    delta0.requires_grad_()

                loss_adv = 0
                for t in range(Config.adversarial_K):
                    # compute grad for delta and parameters of the original model
                    if t == Config.adversarial_K - 1:
                        pgd.restore_grad()  # restore to the grad after the original samples forward
                    else:
                        model.zero_grad()  # clear the grads of the model, delta0's grad is already zero in its initialization

                    if Config.is_pair and not Config.use_pre_train:
                        pred_adv = model.forward_long((batch_embedding + delta0, batch_embedding[1] + delta1),
                                                      input_ori=batch_data)
                    else:
                        if isinstance(model, torch.nn.DataParallel):
                            pred_adv, *_ = model.module.forward_long(batch_embedding + delta0, input_ori=batch_data)
                        else:
                            pred_adv, *_ = model.forward_long(batch_embedding + delta0, input_ori=batch_data)
                    loss_adv = criterion(pred_adv.reshape(-1, pred_adv.shape[-1]), local_labels.reshape(-1))
                    loss_adv.backward()  # add grads of delta and parameters for this step

                    if t == Config.adversarial_K - 1:  # no need to update delta
                        break

                    # update delta according to its grad and epsilon
                    if Config.is_pair and not Config.use_pre_train:
                        delta0 = pgd.attack_on_emb(batch_embedding[0] + delta0, batch_embedding[0],
                                                   delta0.grad, is_first_attack=(t == 0)) - batch_embedding[0]
                        delta1 = pgd.attack_on_emb(batch_embedding[1] + delta1, batch_embedding[1],
                                                   delta1.grad, is_first_attack=(t == 0)) - batch_embedding[1]
                        delta0 = delta0.clone().detach()
                        delta0.requires_grad_()
                        delta1 = delta1.clone().detach()
                        delta1.requires_grad_()
                    else:
                        delta0 = pgd.attack_on_emb(batch_embedding + delta0, batch_embedding,
                                                   delta0.grad, is_first_attack=(t == 0)) - batch_embedding
                        delta0 = delta0.clone().detach()
                        delta0 *= aligned_mask
                        delta0.requires_grad_()
                    if isinstance(model, torch.nn.DataParallel):
                        batch_embedding = model.module.forward_emb(batch_data)
                    else:
                        batch_embedding = model.forward_emb(batch_data)

                # grads of the original model has been added
                train_loss += loss_adv.item()

                optimizer.step()
                scheduler.step()
                model.zero_grad()

                losses.update(loss_adv.data, batch_data['y'].size(0))
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

        accuracy, f1, val_loss, *_ = evaluate(model, criterion, test_generator, returns_metrics=True)
        if best_accu < accuracy:
            best_accu = accuracy
            last_update_epoch = epoch
            torch.save(model.state_dict(), Config.model_save_path)

        resultStr = "mode:{} epoch:{} val_loss:{:.4f} accu:{:.4f}, f1:{}, best accu:{:.4f}, minaccu:{:.4f}".format(
            Config.config_name, epoch, val_loss, accuracy, f1, best_accu, accu_min_train_loss)
        print(resultStr)
    return best_accu


def train_monte_carlo(model, optimizer, scheduler, criterion, train_generator, test_generator):
    pgd = PGD_from_zhengguangyu(model)
    best_accu = 0
    accu_min_train_loss = 0
    last_update_epoch = 0
    for epoch in range(Config.epoch):
        if (Config.early_stop is not None) and (epoch - last_update_epoch > Config.early_stop):
            break

        train_loss = 0
        count = 0
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        bar = Bar('Processing', max=len(train_generator))
        end = time.time()
        with torch.autograd.set_detect_anomaly(True):
            for batch_data0 in tqdm(train_generator, total=len(train_generator), desc='train_pgd'):
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
                local_labels = batch_data['y'].to(Config.device).squeeze()

                data_time.update(time.time() - end)
                count += 1
                model.train()
                optimizer.zero_grad()
                model.zero_grad()

                pred_y, deep_repres, seq_repre = model(batch_data)
                loss = criterion(pred_y.reshape(-1, pred_y.shape[-1]), local_labels.reshape(-1))
                loss.backward()
                pgd.backup_grad()

                aligned_mask = compute_aligned_mask(seq_repre.clone().detach(), Config.aligned_threshold,
                                                    batch_data['x_typeid'])

                if isinstance(model, torch.nn.DataParallel):
                    batch_embedding = model.module.forward_emb(batch_data)
                else:
                    batch_embedding = model.forward_emb(batch_data)

                if Config.is_pair and not Config.use_pre_train:
                    delta0 = torch.zeros_like(batch_embedding[0]).cuda()
                    delta0.requires_grad_()
                    delta1 = torch.zeros_like(batch_embedding[1]).cuda()
                    delta1.requires_grad_()
                else:
                    delta0 = torch.zeros_like(batch_embedding).cuda().uniform_(-1, 1).cuda() * batch_data[
                        'x_mask'].unsqueeze(-1).float()
                    delta0 *= aligned_mask
                    delta0.requires_grad_()

                loss_adv = 0
                for t in range(Config.adversarial_K):
                    # compute grad for delta and parameters of the original model
                    if t == Config.adversarial_K - 1:
                        pgd.restore_grad()  # restore to the grad after the original samples forward
                    else:
                        model.zero_grad()  # clear the grads of the model, delta0's grad is already zero in its initialization

                    if Config.is_pair and not Config.use_pre_train:
                        pred_adv = model.forward_long((batch_embedding + delta0, batch_embedding[1] + delta1),
                                                      input_ori=batch_data)
                    else:
                        if isinstance(model, torch.nn.DataParallel):
                            pred_adv, *_ = model.module.forward_long(batch_embedding + delta0, input_ori=batch_data)
                        else:
                            pred_adv, *_ = model.forward_long(batch_embedding + delta0, input_ori=batch_data)
                    loss_adv = criterion(pred_adv.reshape(-1, pred_adv.shape[-1]), local_labels.reshape(-1))
                    loss_adv.backward()  # add grads of delta and parameters for this step

                    if t == Config.adversarial_K - 1:  # no need to update delta
                        break

                    # update delta according to its grad and epsilon
                    if Config.is_pair and not Config.use_pre_train:
                        delta0 = pgd.attack_on_emb(batch_embedding[0] + delta0, batch_embedding[0],
                                                   delta0.grad, is_first_attack=(t == 0)) - batch_embedding[0]
                        delta1 = pgd.attack_on_emb(batch_embedding[1] + delta1, batch_embedding[1],
                                                   delta1.grad, is_first_attack=(t == 0)) - batch_embedding[1]
                        delta0 = delta0.clone().detach()
                        delta0.requires_grad_()
                        delta1 = delta1.clone().detach()
                        delta1.requires_grad_()
                    else:
                        delta0 = torch.zeros_like(batch_embedding).cuda().uniform_(-1, 1).cuda() * batch_data[
                            'x_mask'].unsqueeze(-1).float()
                        delta0 *= aligned_mask
                        delta0.requires_grad_()
                    if isinstance(model, torch.nn.DataParallel):
                        batch_embedding = model.module.forward_emb(batch_data)
                    else:
                        batch_embedding = model.forward_emb(batch_data)

                # grads of the original model has been added
                train_loss += loss_adv.item()

                optimizer.step()
                scheduler.step()
                model.zero_grad()

                losses.update(loss_adv.data, batch_data['y'].size(0))
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

        accuracy, f1, val_loss, *_ = evaluate(model, criterion, test_generator, True)
        if best_accu < accuracy:
            best_accu = accuracy
            last_update_epoch = epoch
            torch.save(model.state_dict(), Config.model_save_path)

        resultStr = "mode:{} epoch:{} val_loss:{:.4f} accu:{:.4f}, f1:{}, best accu:{:.4f}, minaccu:{:.4f}".format(
            Config.config_name, epoch, val_loss, accuracy, f1, best_accu, accu_min_train_loss)
        print(resultStr)
    return best_accu


# import faiss
import logging
import scipy

logging.basicConfig(filename='./log_polarity.txt', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


class Polarity_Checker:
    def __init__(self, emb_matrix, train_generator):
        self.nn_index = faiss.IndexFlatL2(Config.word_dim)
        self.nn_index.add(emb_matrix.data.cpu().numpy())
        self.word_label = {}
        word_count = {}
        word_label_count = {}
        labels = {}
        for batch_data in tqdm(train_generator, total=len(train_generator), desc='compute data polarity'):
            for i, sent in enumerate(batch_data['x_sent']):
                label = batch_data['y'][i][0].item()
                labels[label] = 1
                for word0 in sent:
                    word = word0.item()
                    if word not in word_count:
                        word_count[word] = 0
                        word_label_count[word] = {}
                    if label not in word_label_count[word]:
                        word_label_count[word][label] = 0
                    word_count[word] += 1
                    word_label_count[word][label] += 1
        for word in word_count:
            fm = 0.0
            self.word_label[word] = np.zeros(len(labels))
            for label in labels:
                fm += word_label_count[word].get(label, 0) + 5.0
            for i, label in enumerate(labels):
                self.word_label[word][i] = 1.0 * (word_label_count[word].get(label, 0) + 5.0) / fm

    def check_polarity(self, batch_embedding, batch_embedding_perturbbed, tokenizer, origin_wordid):
        bsz, seq_len, feat_dim = batch_embedding.size(0), batch_embedding.size(1), batch_embedding.size(2)
        D, I = self.nn_index.search(batch_embedding_perturbbed.view(-1, feat_dim).data.cpu().numpy(), 2)
        word_list = tokenizer.convert_ids_to_tokens(I[:, 0])
        for i in range(bsz):
            logging.warning('perturbed:\t' + ' '.join(word_list[i * seq_len:(i + 1) * seq_len]))
            word_list_ori = tokenizer.convert_ids_to_tokens(origin_wordid[i])
            logging.warning('original :\t' + ' '.join(word_list_ori))
            temp = []
            for j in range(i * seq_len, (i + 1) * seq_len):
                '''if I[j][1] in self.word_label:
                    temp.append('{:.4f}'.format(scipy.stats.entropy(self.word_label[I[j][1]], self.word_label[origin_wordid[i][j - i * seq_len].item()])))
                else:
                    temp.append('{:.4f}'.format(-1.0))'''
                temp.append('{:.4f}'.format(D[j, 0]))
            logging.warning('kl        :\t' + ' '.join(temp))
        return word_list

    def project_word(self, batch_embedding_new, I_prev, emb_matrtix, tokenizer, origin_wordid, write_log=False):
        bsz, seq_len, feat_dim = batch_embedding_new.size(0), batch_embedding_new.size(1), batch_embedding_new.size(2)
        D, I = self.nn_index.search(batch_embedding_new.view(-1, feat_dim).data.cpu().numpy(), 3)
        D = np.concatenate((D, np.zeros_like(D[:, :1])), 1)
        I = np.concatenate((I, np.expand_dims(I_prev, 1)), 1)
        target_I = I[:, 0]
        target_D = D[:, 0]
        for i in range(bsz):
            for j in range(i * seq_len, (i + 1) * seq_len):
                prev_word = tokenizer.convert_ids_to_tokens([I_prev[j]])[0]

                score_max = -1000.0
                index_max = 0
                for k in range(len(D[0])):
                    this_score = 0.0
                    if I[j, k] in self.word_label:
                        kl_k = scipy.stats.entropy(self.word_label[origin_wordid[i][j - i * seq_len].item()],
                                                   self.word_label[I[j, k]])
                    else:
                        kl_k = 0.0
                    if I[j, k] == I_prev[j]:
                        this_score -= 1.0
                    if prev_word.startswith('[') and I[j, k] == I_prev[j]:
                        this_score += 100000
                    if kl_k > 0.1:
                        this_score -= 10.0
                    if this_score > score_max:
                        score_max = this_score
                        index_max = k
                target_I[j] = I[j, index_max]
                target_D[j] = D[j, index_max]

        word_list = tokenizer.convert_ids_to_tokens(target_I)
        if write_log:
            for i in range(bsz):
                word_list_ori = tokenizer.convert_ids_to_tokens(origin_wordid[i])

                dist_list = []
                for j in range(i * seq_len, (i + 1) * seq_len):
                    d = target_D[j]
                    dist_list.append('{:.4f}'.format(d))
                kl_list = []
                for j in range(i * seq_len, (i + 1) * seq_len):
                    if target_I[j] in self.word_label:
                        kl_list.append('{:.4f}'.format(scipy.stats.entropy(self.word_label[
                                                                               origin_wordid[i][
                                                                                   j - i * seq_len].item()],
                                                                           self.word_label[target_I[j]])))
                    else:
                        kl_list.append('{:.4f}'.format(-1.0))
                logging.warning('projected:\t' + ' '.join(word_list[i * seq_len:(i + 1) * seq_len]))
                logging.warning('original :\t' + ' '.join(word_list_ori))
                logging.warning('dis2org  :\t' + ' '.join(dist_list))
                logging.warning('kl2org   :\t' + ' '.join(kl_list))
        return torch.gather(emb_matrtix, 0,
                            torch.from_numpy(I[:, 1]).long().cuda().unsqueeze(-1).expand(bsz * seq_len, feat_dim)).view(
            bsz, seq_len, -1), target_I


def train_pgd_projected(model, optimizer, scheduler, criterion, train_generator, test_generator):
    pgd = PGD_from_zhengguangyu(model)
    polarity = Polarity_Checker(model.module.pre_emb_model.embeddings.word_embeddings.weight, train_generator)
    best_accu = 0
    best_ppl = 1000000
    accu_min_train_loss = 0
    last_update_epoch = 0
    '''projector=MLP(Config.word_dim,100).cuda()
    if Config.multiple_gpu:
        projector = nn.DataParallel(projector)
        projector = projector.cuda()
    else:
        projector = projector.to(Config.device)'''
    # projector_optimizer=optimizers.__dict__[Config.optimizer](model, int(
    #            len(train_generator) / Config.batch_size) * Config.epoch)
    # projector=MLP(Config.word_dim,100)
    for epoch in range(Config.epoch):
        if (Config.early_stop is not None) and (epoch - last_update_epoch > Config.early_stop):
            break

        train_loss = 0
        count = 0
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        bar = Bar('Processing', max=len(train_generator))
        end = time.time()
        with torch.autograd.set_detect_anomaly(True):
            for batch_data0 in tqdm(train_generator, total=len(train_generator), desc='train_pgd_projector'):
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
                local_labels = batch_data['y'].to(Config.device).squeeze()

                data_time.update(time.time() - end)
                count += 1
                model.train()
                optimizer.zero_grad()
                model.zero_grad()

                pred_y = model(batch_data)
                loss = criterion(pred_y.reshape(-1, pred_y.shape[-1]), local_labels.reshape(-1))
                loss.backward()
                pgd.backup_grad()

                if isinstance(model, torch.nn.DataParallel):
                    batch_embedding = model.module.forward_emb(batch_data)
                else:
                    batch_embedding = model.forward_emb(batch_data)

                if Config.is_pair and not Config.use_pre_train:
                    delta0 = torch.zeros_like(batch_embedding[0]).cuda()
                    delta0.requires_grad_()
                    delta1 = torch.zeros_like(batch_embedding[1]).cuda()
                    delta1.requires_grad_()
                else:
                    delta0 = torch.zeros_like(batch_embedding).cuda().uniform_(-1, 1).cuda()
                    delta0 = (pgd.project_emb(batch_embedding + delta0, batch_embedding) - batch_embedding) * \
                             batch_data['x_mask'].unsqueeze(-1).float()
                    delta0 = delta0.detach()
                    delta0.requires_grad_()

                    prev_wordid = batch_data['x_sent'].view(-1).detach().cpu().numpy()

                ##############train model###################
                delta_list = []
                grad_list = {}
                # projector.eval()
                for t in range(Config.adversarial_K):
                    delta_list.append(delta0.clone().unsqueeze(0).detach())
                    # compute grad for delta and parameters of the original model
                    if t == Config.adversarial_K - 1:
                        pgd.restore_grad()  # restore to the grad after the original samples forward
                        break
                    else:
                        model.zero_grad()  # clear the grads of the model, delta0's grad is already zero in its initialization

                    if Config.is_pair and not Config.use_pre_train:
                        pred_adv = model.forward_long((batch_embedding + delta0, batch_embedding[1] + delta1),
                                                      input_ori=batch_data)
                    else:
                        if isinstance(model, torch.nn.DataParallel):
                            # if count==1:
                            # polarity.check_polarity(batch_embedding,batch_embedding+delta0,Config.pre_train_tokenizer,batch_data['x_sent'])
                            pred_adv = model.module.forward_long(batch_embedding + delta0, input_ori=batch_data)
                        else:
                            pred_adv = model.forward_long(batch_embedding + delta0, input_ori=batch_data)
                    loss_adv = criterion(pred_adv.reshape(-1, pred_adv.shape[-1]), local_labels.reshape(-1))
                    loss_adv.backward()  # add grads of delta and parameters for this step

                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            if param.grad is not None:
                                if name not in grad_list:
                                    grad_list[name] = []
                                grad_list[name].append(param.grad.unsqueeze(0).clone().detach())

                    if t == Config.adversarial_K - 1:  # no need to update delta
                        break

                    # update delta according to its grad and epsilon
                    if Config.is_pair and not Config.use_pre_train:
                        delta0 = pgd.attack_on_emb(batch_embedding[0] + delta0, batch_embedding[0],
                                                   delta0.grad, is_first_attack=(t == 0)) - batch_embedding[0]
                        delta1 = pgd.attack_on_emb(batch_embedding[1] + delta1, batch_embedding[1],
                                                   delta1.grad, is_first_attack=(t == 0)) - batch_embedding[1]
                        delta0 = delta0.clone().detach()
                        delta0.requires_grad_()
                        delta1 = delta1.clone().detach()
                        delta1.requires_grad_()
                    else:
                        delta0_prev = delta0.clone().detach()
                        delta0 = pgd.attack_on_emb(batch_embedding + delta0, batch_embedding,
                                                   delta0.grad, is_first_attack=(t == 0),
                                                   wordlevel=False) - batch_embedding
                        delta0, prev_wordid = polarity.project_word(batch_embedding + delta0, prev_wordid,
                                                                    model.module.pre_emb_model.embeddings.word_embeddings.weight,
                                                                    Config.pre_train_tokenizer,
                                                                    batch_data['x_sent'], write_log=(count == 1))
                        delta0 = delta0 - batch_embedding
                        delta0 = delta0 * batch_data['x_mask'].unsqueeze(-1).float()
                        delta0 = delta0.clone().detach()
                        delta0.requires_grad_()
                    if isinstance(model, torch.nn.DataParallel):
                        batch_embedding = model.module.forward_emb(batch_data)
                    else:
                        batch_embedding = model.forward_emb(batch_data)

                '''delta_list=torch.cat(delta_list,dim=0) #k, batch, seq_len, dim
                #k,batch,seq_len,dim_projector
                delta_projected = projector(
                    (delta_list + batch_embedding.unsqueeze(0)).view(-1, batch_embedding.size(-1))).view(
                    Config.adversarial_K, batch_embedding.size(0), batch_embedding.size(1), 100)
                origin_projected = projector(
                    (batch_embedding.unsqueeze(0)).view(-1, batch_embedding.size(-1))).view(
                    1, batch_embedding.size(0), batch_embedding.size(1), 100)
                scores=torch.sum(delta_projected*origin_projected,dim=-1,keepdim=False) #k, batch,seq_len
                scores=F.softmax(scores,dim=0)
                if Config.smoothed:
                    for name, param in model.named_parameters():
                        if name in grad_list:
                            param.grad += torch.sum(torch.cat(grad_list[name], dim=0), dim=0,
                                                    keepdim=False) * scores.unsqueeze(-1)
                else:
                    index=torch.max(scores,dim=0,keepdim=True)[1] #1,batch,seq_len
                    delta0 = torch.gather(delta_list, 0, index.unsqueeze(-1).expand(-1, -1, -1, batch_embedding.size(
                        -1))).squeeze()  # batch,seq_len,dim'''
                pred_adv = model.module.forward_long(batch_embedding + delta0, input_ori=batch_data)
                loss_adv = criterion(pred_adv.reshape(-1, pred_adv.shape[-1]), local_labels.reshape(-1))
                loss_adv.backward()  # add grads of delta and parameters for this step

                # grads of the original model has been added
                train_loss += loss_adv.item()
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                '''
                ##############train projector###################
                projector.train()
                model.eval()
                if isinstance(model, torch.nn.DataParallel):
                    batch_embedding = model.module.forward_emb(batch_data)
                else:
                    batch_embedding = model.forward_emb(batch_data)


                #k,batch,seq_len,dim_projector
                delta_projected = projector(
                    (delta_list + batch_embedding.unsqueeze(0)).view(-1, batch_embedding.size(-1))).view(
                    Config.adversarial_K, batch_embedding.size(0), batch_embedding.size(1), 100)
                origin_projected = projector(
                    (batch_embedding.unsqueeze(0)).view(-1, batch_embedding.size(-1))).view(
                    1, batch_embedding.size(0), batch_embedding.size(1), 100)
                scores=torch.sum(delta_projected*origin_projected,dim=-1,keepdim=False) #k, batch,seq_len
                scores=F.softmax(scores,dim=0)

                if Config.smoothed:
                    for name, param in projector.named_parameters():
                        if name in grad_list:
                            param.grad += torch.sum(torch.cat(grad_list[name], dim=0), dim=0,
                                                    keepdim=False) * scores.unsqueeze(-1)
                else:
                    for i in range(Config.adversarial_K):
                        delta0=delta_list[i]
                        pred_adv = model.module.forward_long(batch_embedding + delta_list[i], input_ori=batch_data)
                        loss_adv = criterion(pred_adv.reshape(-1, pred_adv.shape[-1]), local_labels.reshape(-1))

                        loss_adv.backward()  # add grads of delta and parameters for this step

                projector_optimizer.step()
                projector.zero_grad()'''

                losses.update(loss_adv.data, batch_data['y'].size(0))
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

        accuracy, f1, val_loss, *_ = evaluate(model, criterion, test_generator, True)
        if best_accu < accuracy:
            best_accu = accuracy
            last_update_epoch = epoch
            torch.save(model.state_dict(), Config.model_save_path)

        resultStr = "mode:{} epoch:{} val_loss:{:.4f} accu:{:.4f}, f1:{}, best accu:{:.4f}, minaccu:{:.4f}".format(
            Config.config_name, epoch, val_loss, accuracy, f1, best_accu, accu_min_train_loss)
        print(resultStr)
    return best_accu


def train_freelb(model, optimizer, scheduler, criterion, train_generator, test_generator):
    pgd = PGD_from_zhengguangyu(model)
    best_accu = 0
    best_ppl = 1000000
    accu_min_train_loss = 0
    last_update_epoch = 0
    for epoch in range(Config.epoch):
        if (Config.early_stop is not None) and (epoch - last_update_epoch > Config.early_stop):
            break

        train_loss = 0
        count = 0
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        bar = Bar('Processing', max=len(train_generator))
        end = time.time()
        with torch.autograd.set_detect_anomaly(True):
            for batch_data0 in tqdm(train_generator, total=len(train_generator), desc='train_freelb'):
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
                local_labels = batch_data['y'].to(Config.device).squeeze()

                data_time.update(time.time() - end)
                count += 1
                model.train()
                optimizer.zero_grad()
                model.zero_grad()

                '''pred_y = model(batch_data)
                loss = criterion(pred_y.reshape(-1, pred_y.shape[-1]), local_labels.reshape(-1))
                loss.backward()  #compute gradient via standard training point'''

                if isinstance(model, torch.nn.DataParallel):
                    batch_embedding = model.module.forward_emb(batch_data)
                else:
                    batch_embedding = model.forward_emb(batch_data)

                if Config.is_pair and not Config.use_pre_train:
                    delta0 = torch.zeros_like(batch_embedding[0]).cuda()
                    delta0.requires_grad_()
                    delta1 = torch.zeros_like(batch_embedding[1]).cuda()
                    delta1.requires_grad_()
                else:
                    delta0 = torch.zeros_like(batch_embedding).cuda().uniform_(-1, 1).cuda() * batch_data[
                        'x_mask'].float().unsqueeze(-1)
                    delta0.requires_grad_()

                loss_adv = 0
                for t in range(Config.adversarial_K):
                    # compute grad for delta and parameters of the original model
                    if Config.is_pair and not Config.use_pre_train:
                        pred_adv, *_ = model.forward_long((batch_embedding + delta0, batch_embedding[1] + delta1),
                                                          input_ori=batch_data)
                    else:
                        if isinstance(model, torch.nn.DataParallel):
                            pred_adv, *_ = model.module.forward_long(batch_embedding + delta0, input_ori=batch_data)
                        else:
                            pred_adv, *_ = model.forward_long(batch_embedding + delta0, input_ori=batch_data)
                    loss_adv = criterion(pred_adv.reshape(-1, pred_adv.shape[-1]), local_labels.reshape(-1))
                    loss_adv.backward()

                    if t == Config.adversarial_K - 1:  # no need to update delta
                        break

                    # update delta according to its grad and epsilon
                    if Config.is_pair and not Config.use_pre_train:
                        delta0 = pgd.attack_on_emb(batch_embedding[0] + delta0, batch_embedding[0],
                                                   delta0.grad, is_first_attack=(t == 0)) - batch_embedding[0]
                        delta1 = pgd.attack_on_emb(batch_embedding[1] + delta1, batch_embedding[1],
                                                   delta1.grad, is_first_attack=(t == 0)) - batch_embedding[1]
                        delta0 = delta0.clone().detach()
                        delta0.requires_grad_()
                        delta1 = delta1.clone().detach()
                        delta1.requires_grad_()
                    else:
                        delta0 = pgd.attack_on_emb(batch_embedding + delta0, batch_embedding,
                                                   delta0.grad, is_first_attack=(t == 0)) - batch_embedding
                        delta0 = delta0.clone().detach()
                        delta0.requires_grad_()
                    if isinstance(model, torch.nn.DataParallel):
                        batch_embedding = model.module.forward_emb(batch_data)
                    else:
                        batch_embedding = model.forward_emb(batch_data)

                # grads of the original model has been added
                train_loss += loss_adv.item()
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                losses.update(loss_adv.data, batch_data['y'].size(0))
                if count % 20 == 1:
                    pred_y, *_ = model(batch_data)
                    top1.update(get_acc(pred_y, local_labels), batch_data['y'].size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {accu: .4f}'.format(
                    batch=count + 1, size=len(train_generator), data=data_time.avg, bt=batch_time.avg,
                    total=bar.elapsed_td, eta=bar.eta_td, loss=losses.avg, accu=top1.avg)
                bar.next()
            bar.finish()

        print("epoch:{} train_loss:{}".format(epoch, train_loss / len(train_generator)))

        accuracy, f1, val_loss, *_ = evaluate(model, criterion, test_generator, True)
        if best_accu < accuracy:
            best_accu = accuracy
            last_update_epoch = epoch
            torch.save(model.state_dict(), Config.model_save_path)

        resultStr = "mode:{} epoch:{} val_loss:{:.4f} accu:{:.4f}, f1:{}, best accu:{:.4f}, minaccu:{:.4f}".format(
            Config.config_name, epoch, val_loss, accuracy, f1, best_accu, accu_min_train_loss)
        print(resultStr)
    return best_accu
