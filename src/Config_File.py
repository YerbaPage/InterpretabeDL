# import stanfordnlp
from transformers import *
from Custom_Bert import BertModel_custom
import math
import torch


class Config_base():
    def __init__(self, args):
        self.device = "cuda"
        self.config_name = 'noname'
        self.big_model = 'BigModel'
        self.train_process = args.train_process
        self.databunch_method = args.databunch_method  # 'DataBunch'#'DataBunch_Graph'

        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.config.output_hidden_states = True
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if args.use_custom_bert:
            self.pre_trained_model = BertModel_custom.from_pretrained(
                args.model_name_or_path, config=self.config)
        else:
            self.pre_trained_model = AutoModel.from_pretrained(
                args.model_name_or_path, config=self.config)
            #self.pre_trained_model = (t_model, t_tokenizer, 'bert-base-uncased', t_config)
        self.model_name_or_path = args.model_name_or_path.split('/')[-1]
        #self.output_hidden_states = False
        # self.tokenizer = self.pre_trained_model[1]  # .from_pretrained(self.pre_trained_model[2], do_lower_case=True)
        # self.pre_train_config = self.pre_trained_model[3]  # .from_pretrained(self.pre_trained_model[2])
        #self.pre_train_config.output_hidden_states = True
        self.multiple_gpu = True
        self.do_train = True
        self.do_eval = True
        self.do_test = False

        #self.model = 'bert'

        # training parameters
        self.epoch = 10 if args.epoch is None else args.epoch  # 10
        self.early_stop = 5
        self.continue_train = False
        self.dropout = 0.0
        self.batch_size = args.batch_size
        self.batch_size_test = 32 if args.batch_size_test is None else args.batch_size_test
        # 'ComputeAdamWOptimizer'#'ComputeBertAdamOptimizer'
        self.optimizer = 'ComputeAdamWOptimizer'
        self.clip = 1.0  # 0.25
        self.sent_len = 80

        # dataset setting
        self.task = 'mini-SNLI' if args.dataset is None else args.dataset
        # task_list = ['mini-SNLI', 'MNLI', 'CoLA', 'SST', 'QNLI', 'MSRP']
        # task_list = ['mini-SNLI', 'MNLI', 'QNLI', 'MSRP']
        # task_list = ['SST5-aug-finetuned-GPT2']
        self.dataset_train = self.task
        self.dataset_test = self.task
        file_path = r"../"
        self.is_pair_dict = {'MSRP': True, 'SST': False, 'RTE': True, 'MNLI': True, 'SNLI': True, 'CoLA': False,
                             'QQP': True,
                             'QNLI': True, 'PTB': False, 'PTB_lm': False, 'SST5': False, 'mtl-baby': False,
                             'NER': False,
                             'SRL': False, 'SNLI_visual_representation_learning': False, 'mini-SNLI': True,
                             'SST5-aug-finetuned-GPT2': False, 'e-snli': True, 'e-snli-cause-word': True}
        self.sent_token_dict = {'MSRP': 3, 'SST': 0, 'RTE': 1, 'MNLI': 8, 'SNLI': 7, 'CoLA': 3, 'QQP': 3, 'QNLI': 1,
                                'PTB': 0,
                                'PTB_lm': 0, 'SST5': 0, 'mtl-baby': 1, 'NER': 0, 'SRL': 0,
                                'SNLI_visual_representation_learning': 1, 'mini-SNLI': 5, 'SST5-aug-finetuned-GPT2': 0,
                                'e-snli': 2}
        self.sent2_token_dict = {'MSRP': 4, 'SST': None, 'RTE': 2, 'MNLI': 9, 'SNLI': 8, 'CoLA': None, 'QQP': 4,
                                 'QNLI': 2,
                                 'PTB': None, 'PTB_lm': None, 'SST5': None, 'mtl-baby': False, 'NER': None, 'SRL': None,
                                 'SNLI_visual_representation_learning': None, 'mini-SNLI': 6,
                                 'SST5-aug-finetuned-GPT2': None, 'e-snli': 3, 'e-snli-cause-word': None}
        self.label_token_dict = {'MSRP': 0, 'SST': 1, 'RTE': 3, 'MNLI': 10, 'SNLI': -1, 'CoLA': 1, 'QQP': 5, 'QNLI': 3,
                                 'PTB': 1,
                                 'PTB_lm': None, 'SST5': 1, 'mtl-baby': 0, 'NER': 1,
                                 'SRL': 1, 'SNLI_visual_representation_learning': None, 'mini-SNLI': 0,
                                 'SST5-aug-finetuned-GPT2': 1, 'e-snli': 1}
        self.is_sequence_dict = {'MSRP': False, 'SST': False, 'RTE': False, 'MNLI': False, 'SNLI': False, 'CoLA': False,
                                 'QQP': False, 'QNLI': False, 'PTB': True, 'PTB_lm': True, 'SST5': False,
                                 'mtl-baby': False,
                                 'NER': True, 'SRL': True, 'SNLI_visual_representation_learning': False,
                                 'mini-SNLI': False, 'SST5-aug-finetuned-GPT2': False, 'e-snli': False}
        self.train_file_dict = {'MSRP': file_path + r"data/glue/GLUE-baselines/glue_data/MRPC/train.tsv",
                                'SST': file_path + r"data/SST-2/train.tsv",
                                'RTE': file_path + r"data/glue/GLUE-baselines/glue_data/RTE/train.tsv",
                                'MNLI': file_path + r"data/glue/GLUE-baselines/glue_data/MNLI/train.tsv",
                                'SNLI': file_path + r'data/glue/GLUE-baselines/glue_data/SNLI/train.tsv',
                                'CoLA': file_path + r'data/glue/GLUE-baselines/glue_data/CoLA/train.tsv',
                                'QQP': file_path + r'data/glue/GLUE-baselines/glue_data/QQP/train.tsv',
                                'QNLI': file_path + r'data/glue/GLUE-baselines/glue_data/QNLI/train.tsv',
                                'PTB': file_path + r'data/ptb/train.txt',
                                'PTB_lm': file_path + r'data/penn/train.txt',
                                'SST5': file_path + r'data/sst-5/train.txt',
                                'mtl-baby': file_path + r'data/mtl-dataset/baby.task.train',
                                'NER': file_path + r'data/ner_conll2003/train.txt',
                                'SRL': file_path + r'data/conll05st-release/data/srl/format_train.txt',
                                'SNLI_visual_representation_learning': r'/home/cuiwanyun/robust/src_image/data/flickr30k_train.txt',
                                'mini-SNLI': r'/home/cuiwanyun/robust/src_image/data/mini-snli/snli_1.0_train.txt',
                                'SST5-aug-finetuned-GPT2': file_path + r'data/sst-5-aug-finetuned-GPT2/train.txt',
                                'e-snli': file_path + r'data/e-snli/esnli_train_1.tsv'}
        self.dev_file_dict = {'MSRP': file_path + r"data/glue/GLUE-baselines/glue_data/MRPC/dev.tsv",
                              'SST': file_path + r"data/SST-2/dev.tsv",
                              'RTE': file_path + r"data/glue/GLUE-baselines/glue_data/RTE/dev.tsv",
                              'MNLI': file_path + r"data/glue/GLUE-baselines/glue_data/MNLI/dev_mismatched.tsv",
                              'SNLI': file_path + r'data/glue/GLUE-baselines/glue_data/SNLI/dev.tsv',
                              'CoLA': file_path + r'data/glue/GLUE-baselines/glue_data/CoLA/dev.tsv',
                              'QQP': file_path + r'data/glue/GLUE-baselines/glue_data/QQP/dev.tsv',
                              'QNLI': file_path + r'data/glue/GLUE-baselines/glue_data/QNLI/dev.tsv',
                              'PTB': file_path + r'data/ptb/test.txt',
                              'PTB_lm': file_path + r'data/penn/test.txt',
                              'SST5': file_path + r'data/sst-5/test.txt',
                              'mtl-baby': file_path + r'data/mtl-dataset/baby.task.test',
                              'NER': file_path + r'data/ner_conll2003/test.txt',
                              'SRL': file_path + r'data/conll05st-release/data/srl/format_test.wsj.txt',
                              'SNLI_visual_representation_learning': r'/home/cuiwanyun/robust/src_image/data/flickr30k_test.txt',
                              'mini-SNLI': r'/home/cuiwanyun/robust/src_image/data/mini-snli/snli_1.0_dev.txt',
                              'SST5-aug-finetuned-GPT2': file_path + r'data/sst-5-aug-finetuned-GPT2/dev.txt',
                              'e-snli': file_path + r'data/e-snli/esnli_dev.tsv'}

        # file_path+r'data/model_path/model_{}_{}.h5'.format(dataset,type(self).__name__)
        self.model_save_path = ''

        self.test_file_dict = {'MSRP': file_path + r"data/glue/GLUE-baselines/glue_data/MRPC/test.tsv",
                               'mini-SNLI': file_path + r'data/glue/GLUE-baselines/glue-data/SNLI/text.tsv',
                               'SNLI': file_path + r'data/glue/GLUE-baselines/glue_data/SNLI/test.tsv',
                               'MNLI': file_path + r'data/glue/GLUE-baselines/glue_data/MNLI/test_mismatched.tsv',
                               'RTE': file_path + r"data/glue/GLUE-baselines/glue_data/RTE/test.tsv"}
        self.output_test_file_dict = {'MSRP': file_path + r"data/glue/GLUE-baselines/glue_data/MRPC/test.tsv_predict",
                                      'mini-SNLI': file_path + r'data/glue/GLUE-baselines/glue_data/SNLI/text.tsv_predict',
                                      'SNLI': file_path + r'data/glue/GLUE-baselines/glue_data/SNLI/test.tsv_predict_SNLI',
                                      'MNLI': file_path + r'data/glue/GLUE-baselines/glue_data/MNLI/test.tsv_predict_MNLI',
                                      'RTE': file_path + r"data/glue/GLUE-baselines/glue_data/RTE/test.tsv_predict",
                                      'SST5': file_path + r"data/sst-5/test.tsv_predict",
                                      'SST5-aug-finetuned-GPT2': file_path + r'data/sst-5-aug-finetuned-GPT2/test.txt_predict'}
        self.id_token_dict = {'MSRP': 0, 'mini-SNLI': 0,
                              'RTE': 0, 'SNLI': 0, 'e-snli': 0, 'SST': None}
        pos_tag_count = 50
        self.class_num_dict = {'RTE': 2, 'SST': 2, 'MSRP': 2, 'MNLI': 3, 'SNLI': 3, 'CoLA': 2, 'QQP': 2, 'QNLI': 2,
                               'PTB': pos_tag_count, 'PTB_lm': None, 'SST5': 5, 'mtl-baby': 2, 'NER': 7, 'SRL': 44,
                               'SNLI_visual_representation_learning': 2, 'mini-SNLI': 3, 'SST5-aug-finetuned-GPT2': 5,
                               'e-snli': 3}

        self.output_test_file = None

        self.load_cached_h5 = True
        self.class_num = None  # class_num_dict[dataset_train]
        self.is_pair = None  # is_pair_dict[dataset_train]
        self.is_sequence = None  # is_sequence_dict[dataset_train]

        # process mode
        self.few_count = 10000
        self.load_few = args.load_few

        self.config_name = self.dataset_train + '_' + \
            self.dataset_test + '_' + self.model_name_or_path
        self.class_num = self.class_num_dict[self.dataset_train]
        self.is_pair = self.is_pair_dict[self.dataset_train]
        self.is_sequence = self.is_sequence_dict[self.dataset_train]
        if self.do_test:
            self.output_test_file = self.output_test_file_dict[
                self.dataset_train] + '_' + self.config_name + '_pred.tsv'
        self.model_save_path = self.train_file_dict[self.dataset_train] + \
            '_' + self.config_name + '_pytorch_model.bin'
        self.dataset = self.dataset_train

        self.causal_ratio = args.causal_ratio
        self.grad_clamp = args.grad_clamp
        self.grad_loss_func = args.grad_loss_func
        self.saliancy_method = args.saliancy_method


class Config_relative(Config_base):
    big_model_list = ['BigModel_relative']
    deep_model_list = ['BiLSTM']
    adversarial_mode = 'pgd'
    adversarial_K = 3
    test_mode = False


class Config_relative_embonly(Config_base):
    big_model_list = ['BigModel_relative_embonly']
    deep_model_list = ['Transformer2']


class Config_Gaussian(Config_base):
    PI = 0.5
    SIGMA_1, SIGMA_2 = torch.cuda.FloatTensor(
        [math.exp(-0)]), torch.cuda.FloatTensor([math.exp(-0)])
    big_model_list = ['BigModel_Bayesian']
    gaussian_sample = 3
    prior_method = 'NormLoss'  # 'U_quadratic'  # 'ScaleMixtureGaussian'
    U_quadratic_a = -1
    U_quadratic_b = 1
    prior_factor = 0.0
    variational_factor = 0.0
    bias_method = 'Gaussian'  # 'Gaussian'
    bias_requires_grad = False
    bias_gaussian_low = 0.500
    bias_gaussian_high = 0.5001
    with_gaussian = True


class Config_adversarial_pgd(Config_base):
    adversarial_mode = 'pgd'
    adversarial_K = 3
    epsilon = 0.2
    alpha = 0.1
    epsilon_dict = {'mini-SNLI': 0.2, 'MNLI': 0.2,
                    'CoLA': 0.2, 'SST': 0.2, 'QNLI': 0.15, 'MSRP': 0.4}
    alpha_dict = {'mini-SNLI': 0.1, 'MNLI': 0.1,
                  'CoLA': 0.025, 'SST': 0.1, 'QNLI': 0.1, 'MSRP': 0.04}
    adversarial_K_dict = {'mini-SNLI': 2, 'MNLI': 2,
                          'CoLA': 3, 'SST': 2, 'QNLI': 2, 'MSRP': 3}
    is_adversarial_class = True


class Config_adversarial_pgd_projector(Config_base):
    adversarial_mode = 'pgd_projector'
    adversarial_K = 3
    epsilon = 0.2
    alpha = 0.1
    epsilon_dict = {'mini-SNLI': 1.2, 'MNLI': 0.2,
                    'CoLA': 0.2, 'SST': 1.2, 'QNLI': 0.15, 'MSRP': 0.4}
    alpha_dict = {'mini-SNLI': 0.1, 'MNLI': 0.1,
                  'CoLA': 0.025, 'SST': 0.1, 'QNLI': 0.1, 'MSRP': 0.04}
    adversarial_K_dict = {'mini-SNLI': 2, 'MNLI': 2,
                          'CoLA': 3, 'SST': 3, 'QNLI': 2, 'MSRP': 3}
    smoothed = False
    is_adversarial_class = True


class Config_adversarial_freelb(Config_base):
    adversarial_mode = 'freelb'
    adversarial_K = 3
    epsilon = 0.2
    alpha = 0.1
    epsilon_dict = {'mini-SNLI': 1.2, 'MNLI': 0.2,
                    'CoLA': 0.2, 'SST': 1.2, 'QNLI': 0.15, 'MSRP': 0.4}
    alpha_dict = {'mini-SNLI': 0.1, 'MNLI': 0.1,
                  'CoLA': 0.025, 'SST': 0.1, 'QNLI': 0.1, 'MSRP': 0.04}
    adversarial_K_dict = {'mini-SNLI': 2, 'MNLI': 2,
                          'CoLA': 3, 'SST': 3, 'QNLI': 2, 'MSRP': 3}
    adv_init_mag = 0.1
    use_pre_train_parameters = True
    is_adversarial_class = True


class Config_adversarial_pgd_aligned_mask(Config_base):
    adversarial_mode = 'pgd_aligned_mask'
    adversarial_K = 3
    epsilon = 0.2
    alpha = 0.1
    aligned_threshold = 0.8
    epsilon_dict = {'mini-SNLI': 1.2, 'SNLI': 1.2, 'MNLI': 0.2, 'CoLA': 0.2, 'SST': 1.2, 'QNLI': 0.15, 'MSRP': 0.4,
                    'RTE': 1.2}
    alpha_dict = {'mini-SNLI': 0.3, 'SNLI': 0.3, 'MNLI': 0.1, 'CoLA': 0.025, 'SST': 0.1, 'QNLI': 0.1, 'MSRP': 0.04,
                  'RTE': 0.3}
    adversarial_K_dict = {'mini-SNLI': 2, 'SNLI': 2, 'MNLI': 2,
                          'CoLA': 3, 'SST': 2, 'QNLI': 2, 'MSRP': 3, 'RTE': 2}
    is_adversarial_class = True


class Config_monte_carlo(Config_base):
    adversarial_mode = 'train_monte_carlo'
    adversarial_K = 3
    epsilon = 0.2
    alpha = 0.1
    aligned_threshold = 0.8
    epsilon_dict = {'mini-SNLI': 1.2, 'SNLI': 1.2, 'MNLI': 0.2, 'CoLA': 0.2, 'SST': 1.2, 'QNLI': 0.15, 'MSRP': 0.4,
                    'RTE': 1.2}
    alpha_dict = {'mini-SNLI': 0.3, 'SNLI': 0.3, 'MNLI': 0.1, 'CoLA': 0.025, 'SST': 0.1, 'QNLI': 0.1, 'MSRP': 0.04,
                  'RTE': 0.3}
    adversarial_K_dict = {'mini-SNLI': 2, 'SNLI': 2, 'MNLI': 2,
                          'CoLA': 3, 'SST': 2, 'QNLI': 2, 'MSRP': 3, 'RTE': 2}
    is_adversarial_class = True


class Config_aligned_augment(Config_base):
    train_mode = 'train_aligned_aug'
    use_probase = False
    aug_threshold = 0.8
    hidden_states_aug_layer = 12


class Config_MM_SSL_fine_tuning(Config_base):
    train_mode = 'train_MM_SSL_fine_tuning'
    task_list = ['mini-SNLI']
    # pre_trained_model_path = r'/home/cuiwanyun/robust/lxmert/snap/knowledge distil/BEST_pretrained.pth'
    pre_trained_model_path = r'/home/cuiwanyun/robust/vilbert-multi-task/save/RetrievalFlickr30k_bert_base_6layer_6conect-multi_task_model_knowledge_distil/pytorch_model_knowledge_distil19.bin'
    big_model_list = ['BigModel_two_bert']
    # big_model_list = ['BigModel']


class Config_SNLI_with_Image_Input(Config_base):
    train_mode = 'train'
    task_list = ['SNLI']
    dataset = task_list[0]
    image_folder = r'/home/cuiwanyun/cv_data/flickr30k/flickr30k-images'
    databunch_method = 'DataBunch_with_image_for_joint_embedding'


Config = None


def ComputeConfig(args):
    # global Config
    # Config = globals()[name]
    ret = globals()[args.config]
    ret.use_probase = args.use_probase
    ret.hidden_states_aug_layer = args.hidden_states_aug_layer
    if args.aug_threshold is not None:
        ret.aug_threshold = args.aug_threshold
    if args.aligned_threshold is not None:
        ret.aligned_threshold = args.aligned_threshold
    if args.adversarial_alpha is not None:
        keys = [key for key in ret.alpha_dict]
        for key in keys:
            ret.alpha_dict[key] = args.adversarial_alpha
    if args.adversarial_epsilon is not None:
        keys = [key for key in ret.epsilon_dict]
        for key in keys:
            ret.epsilon_dict[key] = args.adversarial_epsilon
    if args.adversarial_K is not None:
        keys = [key for key in ret.adversarial_K_dict]
        for key in keys:
            ret.adversarial_K_dict[key] = args.adversarial_K
    if args.pre_trained_model_path is not None:
        ret.pre_trained_model_path = args.pre_trained_model_path
    if args.no_use_pre_train_parameters:
        ret.use_pre_train_parameters = False
    if args.batch_size is not None:
        ret.batch_size = args.batch_size
    if args.epoch is not None:
        ret.epoch = args.epoch
    if args.batch_size_test is not None:
        ret.batch_size_test = args.batch_size_test
    if args.pre_trained_model_name is not None:
        if ret.use_pre_train:
            if args.pre_trained_model_name == 'roberta':
                ret.pre_trained_model = (
                    RobertaModel, RobertaTokenizer, 'roberta-base', RobertaConfig)
                ret.pre_train_tokenizer = ret.pre_trained_model[1].from_pretrained(ret.pre_trained_model[2],
                                                                                   do_lower_case=True)
                ret.pre_train_config = ret.pre_trained_model[3].from_pretrained(
                    ret.pre_trained_model[2])
                ret.pre_train_config.output_hidden_states = True

    if args.task_list is not None:
        ret.task_list = args.task_list.split(';')
    ret.test_mode = args.test_mode

    return ret

# Config = Config_Gaussian
# Config=Config_adversarial_pgd
# Config=Config_adversarial_freelb
# Config=Config_adversarial_pgd_projector
# Config=Config_adversarial_pgd_aligned_mask
# Config=Config_aligned_augment
