import csv

def Process_csv2tsv(csv_file):
    tsv_file=csv_file.replace('.csv','.tsv')
    with open(csv_file,'r') as csvin, open(tsv_file, 'w') as tsvout:
        csvin = csv.reader(csvin)
        tsvout = csv.writer(tsvout, delimiter='\t')

        for row in csvin:
            tsvout.writerow(row)

def GenerateExplanations(tsv_file,token_ids,count=None, p_token=None, h_token=None):
    ret=[]
    with open(tsv_file,'r',encoding='utf-8') as reader:
        for index,line in enumerate(reader):
            if index==0:
                continue
            if count is not None and index==count+1:
                break
            tokens=line.strip().split('\t')
            if len(tokens)==0:
                continue
            for id in token_ids:
                if id>=len(tokens):
                    continue
                exp=tokens[id]
                if p_token is not None and (tokens[p_token].lower() in exp.lower()) and (tokens[p_token].lower()!=exp.lower()):
                    #print(exp)
                    #print(tokens[p_token])
                    continue
                if h_token is not None and (tokens[h_token].lower() in exp.lower()) and (tokens[h_token].lower()!=exp.lower()):
                    #print(exp)
                    #print(tokens[h_token])
                    continue
                ret.append(exp)
        reader.close()
    return ret


'''Process_csv2tsv(r'/home/cuiwanyun/nlp_framework/data/e-snli/esnli_train_1.csv')
Process_csv2tsv(r'/home/cuiwanyun/nlp_framework/data/e-snli/esnli_train_2.csv')
Process_csv2tsv(r'/home/cuiwanyun/nlp_framework/data/e-snli/esnli_dev.csv')
Process_csv2tsv(r'/home/cuiwanyun/nlp_framework/data/e-snli/esnli_test.csv')'''

'''explanations = GenerateExplanations(r'../../data/e-snli/esnli_train_1.tsv',[4],10000)
#explanations.extend(GenerateExplanations(r'../../data/e-snli/esnli_train_2.tsv',[4]))
explanations.extend(GenerateExplanations(r'../../data/e-snli/esnli_dev.tsv',[4,9,14]))
explanations.extend(GenerateExplanations(r'../../data/e-snli/esnli_test.tsv',[4,9,14]))
with open(r'../../data/e-snli/esnli_explanations-few.txt','w',encoding='utf-8') as writer:
    for e in explanations:
        writer.write('{}\n'.format(e))
    writer.close()'''

#生成explanation的语料
'''explanations = GenerateExplanations(r'../../data/e-snli/esnli_train_1.tsv',[4],p_token= 2,h_token=3)
explanations.extend(GenerateExplanations(r'../../data/e-snli/esnli_train_2.tsv',[4],p_token= 2,h_token=3))
explanations.extend(GenerateExplanations(r'../../data/e-snli/esnli_dev.tsv',[4,9,14],p_token= 2,h_token=3))
explanations.extend(GenerateExplanations(r'../../data/e-snli/esnli_test.tsv',[4,9,14],p_token= 2,h_token=3))
with open(r'../../data/e-snli/esnli_explanations.txt','w',encoding='utf-8') as writer:
    for e in explanations:
        writer.write('{}\n'.format(e))
    writer.close()'''

'''#生成explanation和premise、hypothesis的语料
explanations = GenerateExplanations(r'../../data/e-snli/esnli_train_1.tsv',[2,3,4],p_token= 2,h_token=3,count=10000)
#explanations.extend(GenerateExplanations(r'../../data/e-snli/esnli_train_2.tsv',[2,3,4],p_token= 2,h_token=3))
explanations.extend(GenerateExplanations(r'../../data/e-snli/esnli_dev.tsv',[2,3,4,9,14],p_token= 2,h_token=3))
explanations.extend(GenerateExplanations(r'../../data/e-snli/esnli_test.tsv',[2,3,4,9,14],p_token= 2,h_token=3))
with open(r'../../data/e-snli/esnli_explanations_p_h_few.txt','w',encoding='utf-8') as writer:
    for e in explanations:
        writer.write('{}\n'.format(e))
    writer.close()'''

#生成premise、hypothesis的语料
explanations = GenerateExplanations(r'../../data/e-snli/esnli_train_1.tsv',[2,3],p_token= 2,h_token=3,count=10000)
#explanations.extend(GenerateExplanations(r'../../data/e-snli/esnli_train_2.tsv',[2,3,4],p_token= 2,h_token=3))
explanations.extend(GenerateExplanations(r'../../data/e-snli/esnli_dev.tsv',[2,3],p_token= 2,h_token=3))
explanations.extend(GenerateExplanations(r'../../data/e-snli/esnli_test.tsv',[2,3],p_token= 2,h_token=3))
with open(r'../../data/e-snli/esnli_p_h_few.txt','w',encoding='utf-8') as writer:
    for e in explanations:
        writer.write('{}\n'.format(e))
    writer.close()