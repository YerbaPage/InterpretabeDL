import torch
import logging
from Config_File import Config

logging.basicConfig(filename='/home/cuiwanyun/nlp_framework/src/log.txt', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('This will get logged to a file')

def ViewTopKNearest(epoch,batch_data,batch_embedding,embedding_matrix,tokenizer,k=50,n=Config.batch_size):
    for i in range(n):
        r = batch_embedding[i:i+1].unsqueeze(-2) - embedding_matrix.unsqueeze(0).unsqueeze(0)
        # batch_size, seq len, vocab size, dim
        r_norm = torch.sum(r.pow(2), dim=-1, keepdim=False).pow(0.5)
        # batch_size, seq len, vocab size
        topk = torch.topk(r_norm, k, dim=-1, largest=False)
        topk_dist = topk[0]
        topk_index = topk[1]
        logging.warning('')
        logging.warning('')
        logging.warning(Config.dataset+' epoch:'+str(epoch)+tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(batch_data['x_sent'][i])))
        for j in range(batch_data['x_sent'].size(1)):
            this_word=tokenizer.convert_ids_to_tokens(batch_data['x_sent'][i][j].item())
            if this_word[0]=='[':
                continue
            topk_word_list=[]
            for kk in range(k):
                topk_word_list.append('{}:{:.3f}'.format(tokenizer.convert_ids_to_tokens(topk_index[0][j][kk].item()),topk_dist[0][j][kk].item()))
            logging.warning(Config.dataset+' epoch:'+str(epoch)+'   '+this_word+'\t'+' \t'.join(topk_word_list))
        logging.warning('\n\n')



