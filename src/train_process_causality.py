import time
import torch
from tqdm import tqdm
from utils import get_acc, TestBIO, transpose_batch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import datetime
from torch.utils.tensorboard import SummaryWriter
from hessian import hessian, jacobian


def compute_saliancy(args, model, batch_data, retain_graph):
    return globals()[args.saliancy_method](args, model, batch_data, retain_graph)

    model.eval()
    pred_y, *_ = model(batch_data)
    ret = []

    for i in range(batch_data['x_sent'].size(0)):
        '''ret_t=[]
        model.zero_grad()
        pred_y_max = torch.max(pred_y[i])
        pred_y_max.backward(retain_graph=True)
        for param in model.parameters():
            for index in batch_data['x_sent'][i]:
                ret_t.append((param.grad.data[index.item()]*param.data[index.item()]).unsqueeze(0).unsqueeze(0))
            break'''

        ret_i = []
        model.zero_grad()
        pred_y_max = torch.max(pred_y[i])
        grad = torch.autograd.grad(pred_y_max, model.parameters(), retain_graph=True)[0]
        for param in model.parameters():
            ret_i = (param.index_select(
                0, batch_data['x_sent'][i])*grad.index_select(0, batch_data['x_sent'][i])).unsqueeze(0)
            break
        #ret.append(torch.cat(ret_i, dim=1))
        ret.append(ret_i)
    temp = torch.sum(pred_y)
    temp.backward()
    ret = torch.cat(ret, dim=0)
    return torch.sum(torch.abs(ret), dim=-1)


def argmax_loss(batch_data, pred_y):
    return torch.sum(torch.max(pred_y, dim=-1)[0])


def argmax_one_loss(batch_data, pred_y):
    ret = torch.max(pred_y, dim=-1)[0]
    ret = ret / ret.detach()  # question
    return torch.sum(ret)


def accuracy_loss(batch_data, pred_y):
    max_value, max_indexes = torch.max(pred_y, dim=-1)
    ret = torch.sum(
        max_value*(max_indexes == batch_data['y']).float().detach())
    return ret


def mae_loss(batch_data, pred_y):
    pred_y = F.softmax(pred_y, dim=-1)
    ret = torch.sum(1.0-(torch.arange(pred_y.size(1)).cuda()
                         == batch_data['y'].cuda())*pred_y)
    return ret


def paired_loss(batch_data, pred_y):
    ce = nn.CrossEntropyLoss()
    y_rand = torch.randint(pred_y.size(1), batch_data['y'].size()).cuda()
    return ce(pred_y, batch_data['y'].cuda())-0.5*ce(pred_y, y_rand)


def compute_saliancy_batch(args, model, batch_data, retain_graph=False):
    model.eval()
    pred_y, *_ = model(batch_data)
    ret = []
    # print(*_)
    model.zero_grad()
    loss = globals()[args.grad_loss_func](batch_data, pred_y)

    # print(model.state_dict()['pre_emb_model.embeddings.word_embeddings.weight'])
    # for name, p in model.named_parameters():
    #     print(name)
    #     print(p)
    # exit()
    # print(model.state_dict()['Trainingmodule.pre_emb_model.embeddings.word_embeddings.weight'])
    # print(hessian(loss, model.state_dict()['Trainingmodule.pre_emb_model.embeddings.word_embeddings.weight']))
    # print(torch.sum(torch.abs(hessian(loss, list(model.named_parameters())[0][1].index_select(0, indexes), allow_unused=True)), -1))
    # exit()

    grad = torch.autograd.grad(loss, model.parameters(), create_graph=retain_graph, retain_graph=True)[0]
    indexes = batch_data['x_sent'].view(-1)  # t
    # print(batch_data['x_sent'])
    # print(torch.abs(hessian(loss, list(model.named_parameters())[0][1][0].index_select(0, indexes), allow_unused=True)))
    # print(list(model.parameters())[0].index_select(0, indexes).data.shape)
    # print(torch.sum(torch.sum(torch.abs(0 != hessian(loss, list(model.parameters())[0].index_select(0, indexes[:200]), allow_unused=True)), -1), -1))
    # print(torch.sum(torch.sum(torch.abs(hessian(loss, list(model.parameters())[0].index_select(0, indexes[:200]), allow_unused=True)), -1), -1))
    # print(torch.sum(torch.sum(torch.abs(hessian(loss, list(model.named_parameters())[0][1].index_select(0, indexes))), -1), -1))
    # exit()
    indexes_count_1 = indexes.unsqueeze(0)
    indexes_count_2 = indexes.unsqueeze(-1)
    indexes_count = torch.sum(((indexes_count_1-indexes_count_2) == 0).float(), -1)
    # print(torch.sum(jacobian(loss, model.parameters()) != 0, -1))
    # print(torch.sum(0 != jacobian(loss, list(model.parameters())[0].index_select(0, indexes)), -1)) 
    # print(list(model.parameters())[0].data.shape)
    # print(indexes)
    # print(torch.sum(0 != jacobian(loss, list(model.parameters())[0]), -1)) # worked

    ############################################
    test_input = list(model.parameters())[0][:] # Failed
    # print(extract_inputs_embeds.data.shape)
    # test_input.retain_grad()
    # test_input = list(model.parameters())[0].index_select(0, indexes) # Failed
    # test_input = list(model.parameters())[0] # Succeeded
    # print(test_input)
    # print(test_input.shape)
    test_out = jacobian(loss, test_input)
    # test_out = hessian(loss, test_input) 
    # print(test_out[test_out != 0]) # worked
    # print('shape: ', test_out.shape)
    ############################################

    # print(torch.sum(0 != jacobian(loss, list(model.parameters())[0].index_select(0, indexes)), -1)) # worked
    # print(hessian(loss, model.parameters())) # worked
    # for param in model.parameters():
    #     for test_param in param:
    #         print(torch.sum(torch.abs(0 != jacobian(pred_y, test_param, create_graph=True)), -1))
        # ret_data = param.data.index_select(0, indexes).view(
        #     batch_data['x_sent'].size(0), batch_data['x_sent'].size(1), -1)
        # ret_grad = grad.index_select(0, indexes).view(
        #     batch_data['x_sent'].size(0), batch_data['x_sent'].size(1), -1)
        # break
    exit()

    ############################################
    ret = ret_data * ret_grad / indexes_count.view(batch_data['x_sent'].size(0), batch_data['x_sent'].size(1), 1)

    temp = torch.sum(pred_y)
    if not retain_graph:
        temp.backward()
        model.zero_grad()
    return torch.sum(torch.abs(ret), dim=-1)


def compute_saliancy_batch_grad(args, model, batch_data, retain_graph=False):
    model.eval()
    pred_y, *_ = model(batch_data)
    ret = []

    model.zero_grad()
    loss = globals()[args.grad_loss_func](batch_data, pred_y)

    grad = torch.autograd.grad(loss, model.parameters(
    ), create_graph=retain_graph, retain_graph=True)[0]
    indexes = batch_data['x_sent'].view(-1)  # t
    indexes_count_1 = indexes.unsqueeze(0)
    indexes_count_2 = indexes.unsqueeze(-1)
    indexes_count = torch.sum(
        ((indexes_count_1-indexes_count_2) == 0).float(), -1)
    for param in model.parameters():
        ret_data = param.data.index_select(0, indexes).view(
            batch_data['x_sent'].size(0), batch_data['x_sent'].size(1), -1)
        ret_grad = grad.index_select(0, indexes).view(
            batch_data['x_sent'].size(0), batch_data['x_sent'].size(1), -1)
        break
    ret = ret_grad / \
        indexes_count.view(batch_data['x_sent'].size(
            0), batch_data['x_sent'].size(1), 1)

    temp = torch.sum(pred_y)
    if not retain_graph:
        temp.backward()
        model.zero_grad()
    return torch.sum(torch.abs(ret), dim=-1)


def compute_hessian_batch(args, model, batch_data, retain_graph=False):
    model.eval()
    pred_y, *_ = model(batch_data)
    ret = []

    model.zero_grad()
    loss = globals()[args.grad_loss_func](batch_data, pred_y)

    grad = torch.autograd.grad(loss, model.parameters(), create_graph=retain_graph, retain_graph=True)[0]
    indexes = batch_data['x_sent'].view(-1)  # t
    indexes_count_1 = indexes.unsqueeze(0)
    indexes_count_2 = indexes.unsqueeze(-1)
    indexes_count = torch.sum(
        ((indexes_count_1-indexes_count_2) == 0).float(), -1)
    for param in model.parameters():
        ret_data = param.data.index_select(0, indexes).view(
            batch_data['x_sent'].size(0), batch_data['x_sent'].size(1), -1)
        ret_grad = grad.index_select(0, indexes).view(
            batch_data['x_sent'].size(0), batch_data['x_sent'].size(1), -1)
        break
    ret = ret_data * ret_grad / indexes_count.view(batch_data['x_sent'].size(0), batch_data['x_sent'].size(1), 1)

    temp = torch.sum(pred_y)
    if not retain_graph:
        temp.backward()
        model.zero_grad()
    return torch.sum(torch.abs(ret), dim=-1)

def visualize(args, epoch, iter, batch_data, word_grad, write_label='a'):
    label = write_label
    prec_fz = 0.0
    prec_fm = 0.0
    with open('log_word_grad_{}.txt'.format(args.dataset), label, encoding='utf-8') as writer:
        writer.write('epoch={}'.format(epoch))
        for i in range(batch_data['x_sent'].size(0)):
            prec_fm += 5.0
            grad_list = []
            grad_word_list = []
            for j in range(batch_data['x_sent'].size(1)):
                word_j = args.tokenizer.convert_ids_to_tokens(
                    batch_data['x_sent'][i][j].item())
                if batch_data['x_sent'][i][j] > 1000 and word_j != '.':
                    grad_list.append(
                        torch.max(torch.abs(word_grad[i][j])).item())
                    grad_word_list.append((grad_list[-1], word_j))
            grad_word_list.sort(key=lambda x: float(x[0]), reverse=True)
            grad_list.sort(reverse=True)
            cause_set = {}
            for j in range(batch_data['x_sent'].size(1)):
                word_j = args.tokenizer.convert_ids_to_tokens(
                    batch_data['x_sent'][i][j].item())
                if word_j == '[PAD]':
                    continue
                if 'cause_mask' in batch_data and batch_data['cause_mask'][i][j] == 1:
                    cause_set[word_j] = 1
                    for k in range(min(5, len(grad_word_list))):
                        if grad_word_list[k][1] == word_j:
                            prec_fz += 1.0
                            break
                    word_j = '*' + word_j + '*'
                grad = torch.max(torch.abs(word_grad[i][j])).item()
                if batch_data['x_sent'][i][j] > 1000 and args.tokenizer.convert_ids_to_tokens(
                        batch_data['x_sent'][i][j].item()) != '.':
                    index = grad_list.index(grad)
                else:
                    index = 'None'
                writer.write('{}:{:.6f},{}  '.format(word_j, grad, index))
            writer.write('\r\n')
            last_word = 'None'
            for j, term in enumerate(grad_word_list):
                if term[1] != last_word:
                    if term[1] in cause_set:
                        this_word = '*' + term[1] + '*'
                    else:
                        this_word = term[1]
                    writer.write('  {}:{},{:.4e}'.format(
                        this_word, j, term[0]))
                last_word = term[1]
            writer.write('\r\n')

        writer.write('\r\n')
        writer.close()
        return prec_fz/prec_fm


def evaluate_causal_word(args, model, criterion, test_generator, count_limit=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    prec5 = AverageMeter()
    top1 = AverageMeter()
    grad_loss = AverageMeter()
    grad0_loss = AverageMeter()
    losses_ce = AverageMeter()
    # print('')
    if count_limit is not None:
        bar = Bar('Testing', max=min(len(test_generator),
                                     int(count_limit/args.batch_size)))
    else:
        bar = Bar('Testing', max=len(test_generator))
    # bar = Bar('Testing', max=len(test_generator))
    # if count_limit is not None:
    #     bar = Bar('Testing', max=min(len(test_generator), int(count_limit/args.batch_size)))
    end = time.time()
    val_loss = 0
    pred_y_all = []
    labels_all = []
    model.train()
    iter = 0

    total_count = 0
    correct_count = 0
    pred_count = 0

    # with torch.no_grad():
    if True:
        for iter, batch_data0 in enumerate(test_generator):
            # batch_data = batch_data0
            batch_data = {}
            for k in batch_data0:
                v = batch_data0[k]
                if isinstance(v, torch.LongTensor) or isinstance(v, torch.FloatTensor):
                    batch_data[k] = v.cuda()
                else:
                    batch_data[k] = v
            if 'cause_mask' in batch_data:
                cause_mask = batch_data['cause_mask']
            else:
                cause_mask = torch.zeros(
                    batch_data['x_sent'].size()).cuda().int()

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
            loss = criterion(pred_y, local_labels)
            val_loss += loss.item()
            losses_ce.update(loss.item(), batch_data['x_sent'].size(0))

            if True:
                loss_gradspred = compute_saliancy(
                    args, model, batch_data, retain_graph=False)
                #loss_gradspred = nn.LayerNorm(loss_gradspred.size()[1:]).cuda()(loss_gradspred)
                '''if iter%max(len(test_generator)//5,1)==0:
                    prec5_this=visualize(args, 'evaluate', iter, batch_data, loss_gradspred)
                    prec5.update(prec5_this,batch_data['x_sent'].size(0))'''
                loss_g0 = torch.sum(loss_gradspred * (1 - cause_mask) * batch_data['x_mask']) / torch.sum(
                    (1 - cause_mask) * batch_data['x_mask'])
                if torch.sum(cause_mask) == 0:
                    loss_g = loss_g0 * 0.0
                else:
                    loss_g = torch.sum(
                        loss_gradspred * cause_mask) / torch.sum(cause_mask)
                grad_loss.update(loss_g.item(), batch_data['x_sent'].size(0))
                grad0_loss.update(loss_g0.item(), batch_data['x_sent'].size(0))
                loss_g *= args.causal_ratio
                loss_g0 *= args.causal_ratio
                if args.grad_clamp:
                    loss += -torch.sum(torch.clamp(loss_gradspred, min=1.0) * cause_mask) / torch.sum(
                        cause_mask)
                else:
                    loss += -loss_g + loss_g0

            losses.update(loss.item(), batch_data['x_sent'].size(0))

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

            # if iter % 20 == 1:
            #     top1.update(get_acc(args, pred_y, local_labels), batch_data['y'].size(0))

            top1.update(get_acc(args, pred_y, local_labels),
                        batch_data['y'].size(0))
            if count_limit is not None and iter*batch_data['y'].size(0) >= count_limit:
                break
            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Batch:{bt:.3f}s |Total:{total:} |ETA:{eta:} |Loss:{loss:.4f} |Loss_ce:{loss_ce:.4f} |Grad:{grad_loss:.4e} |Grad0:{grad0_loss:.4e} |top1:{accu:.4f} |grad_ratio:{ratio:.4f}'.format(
                batch=iter + 1, size=len(test_generator), bt=batch_time.avg, total=bar.elapsed_td,
                eta=bar.eta_td, loss=losses.avg, grad_loss=grad_loss.avg, grad0_loss=grad0_loss.avg, accu=top1.avg, ratio=grad_loss.avg/grad0_loss.avg, loss_ce=losses_ce.avg)
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
            print('NER f1:{}  precision:{}  recall:{}'.format(
                f1, precision, recall))

        return accuracy, f1, val_loss / len(test_generator), grad_loss.avg/grad0_loss.avg


def norm_01(AA):
    size0 = AA.size()
    AA = AA.view(AA.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.view(size0)
    return AA+1e-5


def train_cause_word(args, model, optimizer, scheduler, criterion, train_generator, test_generator):
    best_accu = 0
    best_f1 = 0
    accu_min_train_loss = 0
    last_update_epoch = 0

    train_ratios_log, eval_ratios_log = [], []

    writer_path = '/home/shiyuling/tb/{}_{}_{}_{}'.format(
        args.model_name_or_path, args.batch_size, args.learning_rate, datetime.datetime.now().strftime("%m-%d_%H-%M-%S"))
    writer = SummaryWriter(writer_path)
    print('writer path: {}'.format(writer_path), flush=True)

    global_batch = 0
    for epoch in range(args.epoch):
        if (args.early_stop is not None) and (epoch - last_update_epoch > args.early_stop):
            break
        train_loss = 0
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        ori_losses = AverageMeter()
        top1 = AverageMeter()
        # prec5 = AverageMeter()
        grad_loss = AverageMeter()
        grad0_loss = AverageMeter()
        priores = AverageMeter()
        varies = AverageMeter()
        print('\n')
        bar = Bar('Training', max=len(train_generator))
        end = time.time()
        with torch.autograd.set_detect_anomaly(True):
            for iter, batch_data0 in enumerate(train_generator):
                batch_data = {}
                for k in batch_data0:
                    v = batch_data0[k]
                    if isinstance(v, torch.LongTensor) or isinstance(v, torch.FloatTensor):
                        batch_data[k] = v.cuda()
                    else:
                        batch_data[k] = v
                if 'cause_mask' in batch_data:
                    cause_mask = batch_data['cause_mask']
                else:
                    cause_mask = torch.zeros(
                        batch_data['x_sent'].size()).cuda().int()

                data_time.update(time.time() - end)
                model.train()
                optimizer.zero_grad()
                model.zero_grad()
                loss = 0.0

                if True:
                    loss_gradspred = compute_saliancy(
                        args, model, batch_data, retain_graph=True)
                    #loss_gradspred = nn.LayerNorm(loss_gradspred.size()[1:]).cuda()(loss_gradspred)
                    '''if iter%max(len(train_generator)//5,1)==0:
                        prec5_this = visualize(args, epoch, iter, batch_data, loss_gradspred, write_label='w' if (epoch+iter==0) else 'a')
                        prec5.update(prec5_this,batch_data['x_sent'].size(0))'''
                    loss_g0 = torch.sum(loss_gradspred * (1 - cause_mask) * batch_data['x_mask']) / torch.sum(
                        (1 - cause_mask) * batch_data['x_mask'])  # average without mask
                    if torch.sum(cause_mask) == 0:
                        loss_g = loss_g0 * 0.0
                    else:
                        # average with mask
                        loss_g = torch.sum(
                            loss_gradspred * cause_mask) / torch.sum(cause_mask)
                    grad_loss.update(
                        loss_g.item(), batch_data['x_sent'].size(0))
                    grad0_loss.update(
                        loss_g0.item(), batch_data['x_sent'].size(0))
                    loss_g *= args.causal_ratio
                    loss_g0 *= args.causal_ratio
                    if args.grad_clamp:
                        loss = -torch.sum(torch.clamp(loss_gradspred, min=1.0)
                                          * cause_mask) / torch.sum(cause_mask)
                    else:
                        loss = -loss_g + loss_g0
                        #loss = (-torch.sum(loss_gradspred.pow(0.5) * cause_mask) + torch.sum(loss_gradspred.pow(2) * (1-cause_mask))) *args.causal_ratio
                    #loss = - torch.sum(torch.clamp(torch.abs(loss_gradspred*cause_mask/loss_gradspred_old),max=2.0)) * args.causal_ratio*0.02

                local_labels = batch_data['y'].to(args.device).squeeze()

                pred_y, deep_repre, seq_repre = model(batch_data)
                ce_loss = criterion(
                    pred_y.reshape(-1, pred_y.shape[-1]), local_labels.reshape(-1))
                loss += ce_loss

                loss.backward()

                losses.update(loss.item(), batch_data['x_sent'].size(0))
                ori_losses.update(ce_loss.item(), batch_data['x_sent'].size(0))

                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                train_loss += loss.item()
                optimizer.step()
                scheduler.step()

                top1.update(get_acc(args, pred_y, local_labels),
                            batch_data['y'].size(0))
                # print(top1)

                duration = int(len(train_generator)/10) + 1
                if iter % duration == 0 or iter == len(train_generator)-1:

                    print("")
                    # top1.update(get_acc(args, pred_y, local_labels), batch_data['y'].size(0))

                    # train_accuracy, f1, train_loss, train_ratio = evaluate_causal_word(args, model, criterion, train_generator, count_limit=1000)
                    val_accuracy, f1, val_loss, eval_ratio = evaluate_causal_word(
                        args, model, criterion, test_generator, count_limit=1000)
                    # train_ratios_log.append((train_ratio, train_accuracy, train_loss))
                    eval_ratios_log.append(
                        (eval_ratio, val_accuracy, val_loss))

                    writer.add_scalar('Loss/test', val_loss, global_batch)
                    writer.add_scalar('Ratio/test', eval_ratio, global_batch)
                    writer.add_scalar('Acc/test', val_accuracy, global_batch)

                batch_time.update(time.time() - end)
                end = time.time()

                # bar.suffix = '({batch}/{size}) Batch:{bt:.3f}s |Total:{total:} |ETA:{eta:} |Loss:{loss:.4f} |Loss_ce:{loss_ce:.4f} |Grad:{grad_loss:.4f} |Grad0:{grad0_loss:.4f} |top1:{accu:.4f} |grad_ratio:{ratio:.4f}'.format(
                #     batch=iter + 1, size=len(train_generator), bt=batch_time.avg,
                #     total=bar.elapsed_td, eta=bar.eta_td, loss=losses.avg, grad_loss=grad_loss.avg,
                #     grad0_loss=grad0_loss.avg, accu=top1.avg, ratio=grad_loss.avg/grad0_loss.avg, loss_ce=ori_losses.avg)

                writer.add_scalar('Loss/train', ori_losses.val, global_batch)
                writer.add_scalar('Acc/train', top1.val, global_batch)
                writer.add_scalar('Ratio/train', grad_loss.val /
                                  grad0_loss.val, global_batch)
                writer.add_scalar('Grad_loss/grad0_loss',
                                  grad0_loss.val, global_batch)
                writer.add_scalar('Grad_loss/grad_loss',
                                  grad_loss.val, global_batch)
                writer.flush()
                global_batch += 1

                bar.suffix = '({batch}/{size}) Batch:{bt:.3f}s |Total:{total:} |ETA:{eta:} |Loss:{loss:.4f} |Loss_ce:{loss_ce:.4f} |Grad:{grad_loss:.4f} |Grad0:{grad0_loss:.4f} |top1:{accu:.4f} |grad_ratio:{ratio:.4f}'.format(batch=iter + 1,
                                                                                                                                                                                                                                  size=len(train_generator), bt=batch_time.val, total=bar.elapsed_td, eta=bar.eta_td, loss=losses.val, grad_loss=grad_loss.val, grad0_loss=grad0_loss.val, accu=top1.val, ratio=grad_loss.val/grad0_loss.val, loss_ce=ori_losses.val)
                bar.next()

                # evaluate_causal_word(args, model, criterion, test_generator, True)

            bar.finish()

        print("epoch:{} train_loss:{}".format(
            epoch, train_loss / len(train_generator)))

        accuracy, f1, val_loss, eval_ratio = evaluate_causal_word(
            args, model, criterion, test_generator, None)

        if best_accu < accuracy:
            best_accu = accuracy
            last_update_epoch = epoch
            best_f1 = f1
            torch.save(model.state_dict(), args.model_save_path)
            print('update new model at {}'.format(args.model_save_path))

        resultStr = "mode:{} epoch:{} val_loss:{:.4f} accu:{:.4f}, f1:{}, best accu:{:.4f}, minaccu:{:.4f}".format(
            args.config_name, epoch, val_loss, accuracy, f1, best_accu, accu_min_train_loss)
        print(resultStr)

    with open('ratio{}_log.txt'.format(args.causal_ratio), 'w', encoding='utf-8') as writer:
        for i in range(len(train_ratios_log)):
            writer.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                train_ratios_log[i][0], train_ratios_log[i][1], train_ratios_log[i][2], eval_ratios_log[i][0], eval_ratios_log[i][1], eval_ratios_log[i][2]))
        writer.close()

    if args.class_num == 2:
        return best_accu, best_f1
    else:
        return best_accu
