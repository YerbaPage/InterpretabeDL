def Process(inputFile):
    train_precs=[]
    train_ratios=[]
    eval_precs,eval_ratios=[],[]
    with open('../log/'+inputFile,'r',encoding='utf-8') as reader:
        for line in reader:
            line=line.strip()
            if 'prec@5:' in line:
                index = line.index('prec@5:')
                prec=float(line[index+7:index+7+6])
                index2=line.index('grad_ratio:')
                ratio=float(line[index2+11:index2+11+6])
                if len(train_precs)==len(eval_precs):
                    train_precs.append(prec)
                    train_ratios.append(ratio)
                else:
                    eval_precs.append(prec)
                    eval_ratios.append(ratio)
        reader.close()
    return train_precs,train_ratios,eval_precs,eval_ratios

tag_log={}
for tag in ['log0.01','log0.0']:
    tag_log[tag]=Process(tag)
with open('../log/log_formalized.txt','w',encoding='utf-8') as writer:
    for tag in tag_log:
        writer.write('{}_train_precs\t{}_train_ratios\t{}_eval_precs\t{}_eval_ratios\t'.format(tag,tag,tag,tag))
    writer.write('\n')
    for i in range(min(len(tag_log['log0.01'][0]),len(tag_log['log0.0'][0]))):
        for tag in tag_log:
            for j in range(4):
                writer.write('{}\t'.format(tag_log[tag][j][i]))
        writer.write('\n')
    writer.close()

