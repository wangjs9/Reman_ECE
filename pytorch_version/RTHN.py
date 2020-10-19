import numpy as np
import utils.tf_funcs as func
import torch, sys, time, argparse
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser()
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
# device ##
parser.add_argument('--device', type=torch.device, default='cuda' if torch.cuda.is_available() else 'cpu', required=False, help='device of the model')
# embedding parameters ##
parser.add_argument('--embedding_dim', type=int, default=768, required=False, help='dimension of word embedding')
parser.add_argument('--embedding_dim_pos', type=int, default=50, required=False, help='dimension of position embedding')
# input struct ##
parser.add_argument('--max_doc_len', type=int, default=75, required=False, help='max number of tokens per documents')
parser.add_argument('--max_sen_len', type=int, default=45, required=False, help='max number of tokens per sentence')
# model struct ##
parser.add_argument('--n_hidden', type=int, default=100, required=False, help='number of hidden unit')
parser.add_argument('--n_class', type=int, default=2, required=False, help='number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
parser.add_argument('--log_file_name', type=str, default='', required=False, help='name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
parser.add_argument('--training_iter', type=int, default=15, required=False, help='number of train iter')
parser.add_argument('--scope', type=str, default='RNN', required=False, help='RNN scope')
# not easy to tune , a good posture of using data to train model is very important
parser.add_argument('--batch_size', type=int, default=32, required=False, help='number of example per batch')
parser.add_argument('--lr_assist', type=float, default=0.005, required=False, help='learning rate of assist')
parser.add_argument('--lr_main', type=float, default=0.001, required=False, help='learning rate')
parser.add_argument('--keep_prob1', type=float, default=0.5, required=False, help='word embedding training dropout keep prob')
parser.add_argument('--keep_prob2', type=float, default=1.0, required=False, help='softmax layer dropout keep prob')
parser.add_argument('--l2_reg', type=float, default=1e-5, required=False, help='l2 regularization')
parser.add_argument('--run_times', type=int, default=10, required=False, help='run times of this model')
parser.add_argument('--num_heads', type=int, default=5, required=False, help='the num heads of attention')
parser.add_argument('--n_layers', type=int, default=2, required=False, help='the layers of transformer beside main')
args = parser.parse_args()

def train(config):

    print()

def main():
    train()

if '__name__' == '__main__':
    main()
