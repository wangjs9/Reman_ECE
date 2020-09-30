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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()


def run():
    if args.log_file_name:
        sys.stdout = open(args.log_file_name, 'w')
    # tf.reset_default_graph()
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("***********localtime: ", localtime)
    x_data, y_data, sen_len_data, doc_len_data, word_distance, pos_embedding = func.load_data()

    # word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    pos_embedding = torch.FloatTensor(pos_embedding, device=args.device)
    print('build model...')

    start_time = time.time()
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    y = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    word_dis = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    placeholders = [x, y, sen_len, doc_len, word_dis, keep_prob1, keep_prob2]

    pred, reg, pred_assist_list, reg_assist_list = build_model(x, sen_len, doc_len, word_dis, word_embedding, pos_embedding, keep_prob1, keep_prob2)

    with tf.name_scope('loss'):
        valid_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
        loss_op = - tf.reduce_sum(y * tf.log(pred)) / valid_num + reg * FLAGS.l2_reg
        loss_assist_list = []
        for i in range(FLAGS.n_layers - 1):
            loss_assist = - tf.reduce_sum(y * tf.log(pred_assist_list[i])) / valid_num + reg_assist_list[i] * FLAGS.l2_reg
            loss_assist_list.append(loss_assist)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_main).minimize(loss_op)
        optimizer_assist_list = []
        for i in range(FLAGS.n_layers - 1):
            if i == 0:
                optimizer_assist = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_assist).minimize(loss_assist_list[i])
            else:
                optimizer_assist = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_main).minimize(loss_assist_list[i])
            optimizer_assist_list.append(optimizer_assist)

    true_y_op = tf.argmax(y, 2)
    pred_y_op = tf.argmax(pred, 2)
    pred_y_assist_op_list = []
    for i in range(FLAGS.n_layers - 1):
        pred_y_assist_op = tf.argmax(pred_assist_list[i], 2)
        pred_y_assist_op_list.append(pred_y_assist_op)

    print('build model done!\n')
    prob_list_pr, y_label = [], []
    # Training Code Block
    print_training_info()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        kf, fold, SID = KFold(n_splits=10), 1, 0
        Id = []
        p_list, r_list, f1_list = [], [], []
        for train, test in kf.split(x_data):
            tr_x, tr_y, tr_sen_len, tr_doc_len, tr_word_dis = map(lambda x: x[train],
                [x_data, y_data, sen_len_data, doc_len_data, word_distance])
            te_x, te_y, te_sen_len, te_doc_len, te_word_dis = map(lambda x: x[test],
                [x_data, y_data, sen_len_data, doc_len_data, word_distance])

            precision_list, recall_list, FF1_list = [], [], []
            pre_list, true_list, pre_list_prob = [], [], []

            sess.run(tf.global_variables_initializer())
            print('############# fold {} ###############'.format(fold))
            fold += 1
            max_f1 = 0.0
            print('train docs: {}    test docs: {}'.format(len(tr_y), len(te_y)))

            '''*********GP*********'''
            for layer in range(FLAGS.n_layers - 1):
                if layer == 0:
                    training_iter = FLAGS.training_iter
                else:
                    training_iter = FLAGS.training_iter - 5
                for i in range(training_iter):
                    step = 1
                    for train, _ in get_batch_data(tr_x, tr_y, tr_sen_len, tr_doc_len, tr_word_dis, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.batch_size):
                        _, loss, pred_y, true_y, pred_prob, doc_len_batch = sess.run(
                            [optimizer_assist_list[layer], loss_assist_list[layer], pred_y_assist_op_list[layer], true_y_op, pred_assist_list[layer], doc_len],
                            feed_dict=dict(zip(placeholders, train)))
                        acc_assist, p_assist, r_assist, f1_assist = func.acc_prf(pred_y, true_y, doc_len_batch)
                        if step % 10 == 0:
                            print('GL{}: epoch {}: step {}: loss {:.4f} acc {:.4f}'.format(layer + 1, i + 1, step, loss, acc_assist))
                        step = step + 1

            '''*********Train********'''
            for epoch in range(FLAGS.training_iter):
                step = 1
                for train, _ in get_batch_data(tr_x, tr_y, tr_sen_len, tr_doc_len, tr_word_dis, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.batch_size):
                    _, loss, pred_y, true_y, pred_prob, doc_len_batch = sess.run(
                        [optimizer, loss_op, pred_y_op, true_y_op, pred, doc_len],
                        feed_dict=dict(zip(placeholders, train)))
                    acc, p, r, f1 = func.acc_prf(pred_y, true_y, doc_len_batch)
                    if step % 5 == 0:
                        print('epoch {}: step {}: loss {:.4f} acc {:.4f}'.format(epoch + 1, step, loss, acc))
                    step = step + 1

                '''*********Test********'''
                test = [te_x, te_y, te_sen_len, te_doc_len, te_word_dis, 1., 1.]
                loss, pred_y, true_y, pred_prob = sess.run(
                    [loss_op, pred_y_op, true_y_op, pred], feed_dict=dict(zip(placeholders, test)))

                end_time = time.time()

                true_list.append(true_y)
                pre_list.append(pred_y)
                pre_list_prob.append(pred_prob)

                acc, p, r, f1 = func.acc_prf(pred_y, true_y, te_doc_len)
                precision_list.append(p)
                recall_list.append(r)
                FF1_list.append(f1)
                if f1 > max_f1:
                    max_acc, max_p, max_r, max_f1 = acc, p, r, f1
                print('\ntest: epoch {}: loss {:.4f} acc {:.4f}\np: {:.4f} r: {:.4f} f1: {:.4f} max_f1 {:.4f}\n'.format(
                    epoch + 1, loss, acc, p, r, f1, max_f1))

            Id.append(len(te_x))
            SID = np.sum(Id) - len(te_x)
            _, maxIndex = func.maxS(FF1_list)
            print("maxIndex:", maxIndex)
            print('Optimization Finished!\n')
            pred_prob = pre_list_prob[maxIndex]

            for i in range(pred_y.shape[0]):
                for j in range(te_doc_len[i]):
                    prob_list_pr.append(pred_prob[i][j][1])
                    y_label.append(true_y[i][j])

            p_list.append(max_p)
            r_list.append(max_r)
            f1_list.append(max_f1)
        print("running time: ", str((end_time - start_time) / 60.))
        print_training_info()
        p, r, f1 = map(lambda x: np.array(x).mean(), [p_list, r_list, f1_list])

        print("f1_score in 10 fold: {}\naverage : {} {} {}\n".format(np.array(f1_list).reshape(-1, 1), round(p, 4), round(r, 4), round(f1, 4)))
        return p, r, f1

def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, learning_rate-{}, keep_prob1-{}, num_heads-{}, n_layers-{}'.format(
        args.batch_size, args.lr_main, args.keep_prob1, args.num_heads, args.n_layers))
    print('training_iter-{}, scope-{}\n'.format(args.training_iter, args.scope))

def get_batch_data(x, y, sen_len, doc_len, word_dis, keep_prob1, keep_prob2, batch_size, test=False):
    for index in func.batch_index(len(y), batch_size, test):
        feed_list = [x[index], y[index], sen_len[index], doc_len[index], word_dis[index], keep_prob1, keep_prob2]
        yield feed_list, len(index)

def senEncode_softmax(s_senEncode, w_varible, b_varible, n_feature, doc_len):
    s = tf.reshape(s_senEncode, [-1, n_feature])
    s = tf.nn.dropout(s, keep_prob=args.keep_prob2)
    w = func.get_weight_varible(w_varible, [n_feature, args.n_class])
    b = func.get_weight_varible(b_varible, [args.n_class])
    pred = tf.matmul(s, w) + b
    pred *= func.getmask(doc_len, args.max_doc_len, [-1, 1])
    pred = tf.nn.softmax(pred)
    pred = tf.reshape(pred, [-1, args.max_doc_len, args.n_class])
    reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    return pred, reg

def trans_func(senEncode_dis, senEncode, n_feature, out_units, scope_var):
    senEncode_assist = trans.multihead_attention(queries=senEncode_dis,
                                            keys=senEncode_dis,
                                            values=senEncode,
                                            units_query=n_feature,
                                            num_heads=args.num_heads,
                                            dropout_rate=0,
                                            is_training=True,
                                            scope=scope_var)
    senEncode_assist = trans.feedforward_1(senEncode_assist, n_feature, out_units)
    return senEncode_assist

def main(_):
    grid_search = {}
    params = {"n_layers": [4, 5]}

    params_search = list(ParameterGrid(params))

    for i, param in enumerate(params_search):
        print("*************params_search_{}*************".format(i + 1))
        print(param)
        for key, value in param.items():
            setattr(args, key, value)
            print(args.n_layers)
        p_list, r_list, f1_list = [], [], []
        for i in range(args.run_times):
            print("*************run(){}*************".format(i + 1))
            p, r, f1 = run()
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)

        for i in range(args.run_times):
            print(round(p_list[i], 4), round(r_list[i], 4), round(f1_list[i], 4))
        print("avg_prf: ", np.mean(p_list), np.mean(r_list), np.mean(f1_list))

        grid_search[str(param)] = {"PRF": [round(np.mean(p_list), 4), round(np.mean(r_list), 4), round(np.mean(f1_list), 4)]}

    for key, value in grid_search.items():
        print("Main: ", key, value)

if __name__ == '__main__':
    main('')
