import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

def getmask(length, max_len, out_shape):
    # return torch.reshape(ret, out_shape)
    print('To be continue')

class Config(object):
    def __init__(self):
        # device ##
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # embedding parameters ##
        self.embedding_dim = 768 # dimension of word embedding
        self.embedding_dim_pos = 50 # dimension of position embedding
        # input struct ##
        self.max_doc_len = 75 # max number of tokens per documents
        self.max_sen_len = 45 # max number of tokens per sentence
        # model struct ##
        self.n_hidden = 100 # number of hidden unit
        self.n_class = 2 # number of distinct class
        # >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
        self.log_file_name = '' # name of log file
        # >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
        self.training_iter = 15 # number of train iter
        self.scope = 'RNN' # RNN scope
        # not easy to tune , a good posture of using data to train model is very important
        self.batch_size = 32 # number of example per batch
        self.lr_assist = 0.005 # learning rate of assist
        self.lr_main = 0.001 # learning rate
        self.keep_prob1 = 0.5 # word embedding training dropout keep prob
        self.keep_prob2 = 1.0 # softmax layer dropout keep prob
        self.l2_reg = 1e-5 # l2 regularization
        self.run_times = 10 # run times of this model
        self.num_heads = 5 # the num heads of attention
        self.n_layers = 2 # the layers of transformer beside main


class Model(nn.Module):
    def __init__(self, config, pos_emebedding, dropout=0.5):
        super(Model, self).__init__()
        self.device = config.device
        ### Clasue Level
        self.embedding_dim = config.embedding_dim
        self.embedding_dim_pos = config.embedding_dim_pos
        self.max_sen_len = config.max_sen_len
        self.max_doc_len = config.max_doc_len
        self.n_hidden = config.n_hidden
        self.pos_emebedding = nn.Embedding.from_pretrained(pos_emebedding, freeze=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.encoder = BertModel.from_pretrained('bert-base-uncased',output_attentions=False,
                    output_hidden_states=False)
        self.biLSTM = nn.LSTM(config.embedding_dim, config.n_hidden, batch_first=True, bidirectional=True, dropout=dropout)
        self.att_var = Attention(config.n_hidden * 2)
        ### Document Level
        self.n_feature = 2 * config.n_hidden + config.embedding_dim_pos
        self.out_units = 2 * config.n_hidden
        self.n_layers = config.n_layers
        '''*******GL1******'''
        if config.n_layers > 1:
            self.multihead_attention = nn.MultiheadAttention(self.n_feature, config.num_heads)
            self.ffd = nn.Linear(self.max_doc_len, self.max_doc_len, bias=True)

    def forward(self, x, x_mask, y, sen_len, doc_len, word_dis):
        """
        :param x: (batch_size, max_doc_len, max_sen_len, embedding_size)
        :param x_mask: (batch_size, max_doc_len, max_sen_len)
        :param y: (batch_size, max_doc_len, 2)
        :param sen_len: (batch_size, max_doc_len)
        :param doc_len: (batch_size, )
        :param word_dis: (batch_size, max_doc_len, max_sen_len)
        :return:
        """
        ########################
        ### word level embedding:
        ########################
        x = self.encoder(x, token_type_ids=None, attention_mask=x_mask).reshape((-1, self.max_sen_len, self.n_hidden))
        # x: (batch_size*max_doc_len, max_sen_len, num_direction * hidden_size)
        wordEncode, _  = self.biLSTM(x)
        # wordEncode: (batch_size*max_doc_len, max_sen_len, 2 * hidden_size)
        ##--------------------##
        ########################
        ### sentence level embedding:
        ########################
        sen_len = torch.reshape(sen_len, (-1))
        x_mask = torch.reshape(x_mask, (-1, self.max_sen_len))
        # sen_len: (batch_size*max_doc_len, )
        senEncode = self.att_var(wordEncode, sen_len, x_mask).reshape((-1, self.max_doc_len, 2 * self.n_hidden))
        # senEncode: (batch_size*max_doc_len, 2 * hidden_size)
        batch_size = senEncode.shape[0]
        word_dis = self.pos_emebedding(word_dis)
        word_dis = word_dis[:, :, 0, :].reshape((-1, self.max_doc_len, self.embedding_dim_pos))
        senEncode_dis = torch.concat((senEncode, word_dis), axis=2)
        # senEncode_dis: (batch_size, max_doc_len, 2 * hidden_size + embedding_dim_pos)
        pred_zeros = torch.zeros((batch_size, self.max_doc_len, self.max_doc_len), device=self.device)
        pred_ones = torch.ones_like(pred_zeros, device=self.device)
        pred_two = torch.Tensor(batch_size, self.max_doc_len, self.max_doc_len).fill_(2).to(self.device)
        matrix = (1 - torch.eye(self.max_doc_len, device=self.device)).reshape(1, self.max_doc_len, self.max_doc_len) + pred_zeros
        pred_assist_list, reg_assist_list, pred_assist_label_list = [], [], []
        if self.n_layer > 1:
            '''*******GL1******'''
            senEncode = self.multihead_attention(senEncode_dis, senEncode_dis, senEncode)

        '''*******GL n******'''
        for i in range(2, self.n_layers):
            # senEncode_assist = torch.cat((senEncode, pred_assist_label), axis=2)
            print('To be continue ...')

class encode_softmax(nn.Module):
    def __init__(self, n_feature, n_class):
        self.n_feature = n_feature
        self.n_class = n_class
        self.linear = nn.Linear(n_feature, n_class)
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, senEncode, doc_len):
        s = torch.reshape(senEncode, (-1, self.n_feature))
        pred = self.linear(s) # * mask
        #
        pred = self.soft_max(pred)



class Attention(nn.Module):
    def __init__(self, dim):
        self.dim = dim
        self.att_u = nn.Tanh(nn.Linear(dim, dim, bias=True))
        self.att_alpha = nn.Linear(dim, 1, bias=False)

    def forward(self, inputs, mask):
        """
        :param inputs: (batch_size*max_doc_len, max_sen_len, hidden_size)
        :param mask: (batch_size*max_doc_len, max_sen_len)
        :return: (batch_size, hidden_size)
        """
        max_sen_len = inputs.shape[1]
        tmp = inputs.reshape((-1, self.dim))
        # (batch_size*max_doc_len*max_sen_len, hidden_size)
        u = self.att_u(tmp)
        # (batch_size*max_doc_len*max_sen_len, hidden_size)
        alpha = torch.reshape(self.att_alpha(u), (-1, 1, max_sen_len))
        # (batch_size*max_doc_len, 1, max_sen_len)

        ### max_soft by length
        origin_size = alpha.shape
        alpha = torch.exp(alpha) * mask.unsqueeze(1)
        alpha = alpha.reshape(origin_size)
        # alpha: (batch_size*max_doc_len, 1, max_sen_len)
        _sum = torch.sum(alpha, 2, keepdim=True) + 1e-9
        alpha = torch.true_divide(alpha, _sum)

        return torch.matmul(alpha, inputs).reshape((-1, self.dim))





