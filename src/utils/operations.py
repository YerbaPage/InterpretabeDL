import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def TestBIO(pred_i,label_i,i_id=1,o_id=0):
    if pred_i.size()[0]<=1:
        return 0,0,0
    total_count=0
    correct_count=0
    pred_count=0
    pred=torch.argmax(pred_i,2).detach().cpu().numpy()
    label=label_i.detach().cpu().numpy()

    this_pred=""
    this_label=""
    for i in range(len(pred)):
        labelset = {}
        predset = {}
        for j in range(len(pred[0])):
            if pred[i][j]!=i_id:
                if this_pred!="":
                    predset[this_pred]=1
                    this_pred=""
            if label[i][j]!=i_id:
                if this_label != "":
                    labelset[this_label] = 1
                    this_label=""
            if label[i][j]==Config.class_num-1:
                break

            if pred[i][j]!=o_id and pred[i][j]!=Config.class_num-1:
                this_pred+=str(j)+str(pred[i][j])

            if label[i][j] != o_id and label[i][j] != Config.class_num - 1:
                this_label+= str(j) + str(label[i][j])
        total_count+=len(labelset)
        pred_count+=len(predset)
        for t in predset:
            if t in labelset:
                correct_count+=1

    return total_count,pred_count,correct_count



def get_acc(args,pred_y,local_labels):
    pred_y_this=[]
    labels_this=[]
    _, pred_y_label = torch.max(pred_y, -1)
    if args.is_sequence:
        pred_y_label=pred_y_label.view(-1)
        local_labels=local_labels.contiguous().view(-1)
    #    pred_y=torch.argmax(pred_y,-1,keepdim=False)

    for i, t in enumerate(pred_y_label):
        if args.is_sequence:
            if local_labels[i] != args.pos_tag_count - 1:
                pred_y_this.append(pred_y_label[i].item())
                labels_this.append(local_labels[i].item())
        else:
            pred_y_this.append(t.item())
            labels_this.append(local_labels[i].item())
    return accuracy_score(labels_this,pred_y_this)

def transpose_batch(batch_data):
    ret={}
    for key in batch_data:
        ret[key]=batch_data[key].t()
    return ret