train_id_file=""
val_id_file=""
test_id_file=""
import os
import shutil

def ReadIdSet(input_id_file,input_image_folder,input_text_file,output_image_foler,output_text_file):
    ids={}
    with open(input_id_file) as reader:
        for line in reader:
            tokens=line.strip().split('\t')
            id=tokens[1][:-6]
            ids[id]=1
        reader.close()
    for file in os.listdir(input_image_folder):
        if file.endswith(".jpg"):
            id=file[:-4]
            #print(id,file)
            if id in ids:
                tgt_file_path = os.path.join(output_image_foler, file)
                src_file_path = os.path.join(input_image_folder, file)
                shutil.copyfile(src_file_path,tgt_file_path)
            else:
                print(id)
    with open(input_text_file,'r',encoding='utf-8') as text_reader:
        with open(output_text_file,'w',encoding='utf-8') as text_writer:
            for line in text_reader:
                id=line.split('#')[0][:-4]
                if id in ids:
                    text_writer.write('{}\n'.format(line.strip()))
            text_writer.close()
        text_reader.close()


ReadIdSet(r'/home/cuiwanyun/nlp_framework/data/glue/GLUE-baselines/glue_data/SNLI/train.tsv',
          r'/home/cuiwanyun/robust/src_image/data/flickr30k-images/',
          r'/home/cuiwanyun/robust/src_image/data/flickr30k/results_20130124.token',
          r'/home/cuiwanyun/robust/src_image/data/flickr30k_train', r'/home/cuiwanyun/robust/src_image/data/flickr30k_train.txt')

ReadIdSet(r'/home/cuiwanyun/nlp_framework/data/glue/GLUE-baselines/glue_data/SNLI/dev.tsv',
          r'/home/cuiwanyun/robust/src_image/data/flickr30k-images/',
          r'/home/cuiwanyun/robust/src_image/data/flickr30k/results_20130124.token',
          r'/home/cuiwanyun/robust/src_image/data/flickr30k_dev', r'/home/cuiwanyun/robust/src_image/data/flickr30k_dev.txt')

ReadIdSet(r'/home/cuiwanyun/nlp_framework/data/glue/GLUE-baselines/glue_data/SNLI/test.tsv',
          r'/home/cuiwanyun/robust/src_image/data/flickr30k-images/',
          r'/home/cuiwanyun/robust/src_image/data/flickr30k/results_20130124.token',
          r'/home/cuiwanyun/robust/src_image/data/flickr30k_test', r'/home/cuiwanyun/robust/src_image/data/flickr30k_test.txt')
