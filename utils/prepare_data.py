import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import re, sys, gc
import pickle as pk
# path = '../reman/'
path = './empatheticdialogues/'
max_doc_len = 35
max_sen_len = 50

def process_data(text_file, input_file, output_file, context=False):
    """
    :param text_file: the file including context.
    :param input_file: the file including emotion label, cause spans, and emotion spans.
    :param output_file: output file.
    :return: None. A new file named output_file is generated.
    """
    doctext = pd.read_csv(text_file, sep='\t', header=None, index_col=False, encoding='UTF-8')
    ece = pd.read_csv(input_file, sep='\t', header=0, index_col=False, encoding='UTF-8')

    punc = r'[!",-.:;?~]\s'

    if context:
        col_name = ['conv_id', 'no', 'context', 'emotion_clause', 'cause_clause', 'label', 'relative_pos', 'cause', 'clause']
    else:
        col_name = ['conv_id', 'no', 'label', 'relative_pos', 'cause', 'clause']
    data = []
    SenID = 0
    for conv_id, row in ece.iterrows():
        text = doctext.loc[conv_id,0]
        text_tokens = re.split(punc, text)
        lengths = [len(tokens) for tokens in text_tokens]
        interval = [0]

        for i, length in enumerate(lengths[:-1]):
            interval.append(length+interval[i])

        emotion_label, emotion_spans, cause_spans = row
        emotion_label = emotion_label[1:-1].replace('\'', '').split(',')
        emotion_spans = emotion_spans[1:-1].replace('\'', '').split(',')
        cause_spans = cause_spans[1:-1].replace('\'', '').split(',')
        for no, label in enumerate(emotion_label):
            emotion_span1, emotion_span2 = int(emotion_spans[no*2]), int(emotion_spans[no*2+1])
            cause_span1, cause_span2 = int(cause_spans[no*2]), int(cause_spans[no*2+1])
            # No need to delete, the clauses of emotion and cause
            emotion_clause = text[emotion_span1:emotion_span2]
            cause_clause = text[cause_span1:cause_span2]
            emotion_id = 0
            cause_id = 0
            for length_id in range(len(interval)):
                if emotion_span1 > interval[length_id]:
                    emotion_id = length_id
                if cause_span1 > interval[length_id]:
                    cause_id = length_id
            for clause_id, clause in enumerate(text_tokens):
                if context:
                    temp = [SenID, clause_id, text, emotion_clause, cause_clause, label, clause_id-emotion_id, clause_id==cause_id, clause]
                else:
                    temp = [SenID, clause_id, label, clause_id-emotion_id, clause_id==cause_id, clause]
                data.append(temp)
            SenID += 1

    data = pd.DataFrame(data)
    data.to_csv(output_file, encoding='UTF-8', sep='\t', index=False, header=col_name)

def load_w2v(embedding_dim_pos):
    """
    :param embedding_dim_pos:
    :return:
    """
    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(
        loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(-68, 34)])
    # embedding.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim)) for i in range(-68,34)])
    embedding_pos = np.array(embedding_pos)
    pk.dump(embedding_pos, open(path + 'embedding_pos.txt', 'wb'))
    print("embedding_pos.shape: {}".format(embedding_pos.shape))
    return embedding_pos

def load_data(input_file, max_doc_len=max_doc_len, max_sen_len=max_sen_len):
    """
    :param input_file:
    :param max_doc_len: the max length of
    :param max_sen_len: the max number of word in a sentence
    :return:
    """
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print('load data...')
    relative_pos, x, y, sen_len, doc_len = [], [], [], [], []

    y_clause_cause, clause_all, tmp_clause_len, relative_pos_all = np.zeros((max_doc_len, 2)), [], [], []
    next_ID = 1
    n_clause, yes_clause, no_clause, n_cut = [0] * 4

    bert = BertModel.from_pretrained('bert-base-uncased',output_attentions=False,
                    output_hidden_states=False)
    for param in bert.parameters():
        param.requires_grad = False

    data = pd.read_csv(input_file, sep='\t', encoding='UTF-8', header=0)
    for index, line in data.iterrows():
        n_clause += 1
        senID, clause_idx, emo_word, sen_pos, cause, words = line
        word_pos = int(sen_pos) + 69

        if next_ID == senID:
            doc_len.append(len(clause_all))

            for j in range(max_doc_len - len(clause_all)):
                clause_all.append(np.zeros((max_sen_len, 768)))
                tmp_clause_len.append(0)
                relative_pos_all.append(np.zeros((max_sen_len,)))
            relative_pos.append(relative_pos_all)
            x.append(clause_all)
            y.append(y_clause_cause)
            sen_len.append(tmp_clause_len)
            y_clause_cause, clause_all, tmp_clause_len, relative_pos_all = np.zeros((max_doc_len, 2)), [], [], []
            next_ID = senID + 1

        encoded_dict = tokenizer.encode_plus(words,
                             add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                             max_length=max_sen_len,  # Pad & truncate all sentences.
                             pad_to_max_length=True,
                             return_attention_mask=True,  # Construct attn. masks.
                             return_tensors='pt',  # Return pytorch tensors.
                             truncation_strategy='longest_first')

        clause, _ = bert(encoded_dict['input_ids'], token_type_ids=None, attention_mask=encoded_dict['attention_mask'])
        relative_pos_clause = [word_pos] * max_sen_len
        relative_pos_all.append(np.array(relative_pos_clause))
        clause_all.append(clause.cpu().numpy()[0])
        tmp_clause_len.append(sum(encoded_dict['attention_mask'][0]))
        if cause:
            no_clause += 1
            y_clause_cause[clause_idx] = [1,0]
        else:
            yes_clause += 1
            y_clause_cause[clause_idx] = [0,1]

    relative_pos, x, y, sen_len, doc_len = map(np.array, [relative_pos, x, y, sen_len, doc_len])
    pk.dump(relative_pos, open(path + 'relative_pos.txt', 'wb'))
    pk.dump(x, open(path + 'x.txt', 'wb'))
    pk.dump(y, open(path + 'y.txt', 'wb'))
    pk.dump(sen_len, open(path + 'sen_len.txt', 'wb'))
    pk.dump(doc_len, open(path + 'doc_len.txt', 'wb'))

    print('relative_pos.shape {}\nx.shape {} \ny.shape {} \nsen_len.shape {} \ndoc_len.shape {}\n'.format(
        relative_pos.shape, x.shape, y.shape, sen_len.shape, doc_len.shape
    ))
    print('n_clause {}, yes_clause {}, no_clause {}, n_cut {}'.format(n_clause, yes_clause, no_clause, n_cut))
    print('load data done!\n')
    return x, y, sen_len, doc_len

def load_test_data(input_file, round, up_id, bottom_id, max_doc_len=max_doc_len, max_sen_len=max_sen_len):
    # gc.set_debug()#gc.DEBUG_STATS|gc.DEBUG_LEAK)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print('load data...')
    relative_pos, x, y, sen_len, doc_len = [], [], [], [], []
    y_clause_cause, clause_all, tmp_clause_len, relative_pos_all = np.zeros((max_doc_len, 2)), [], [], []
    index_list = []
    next_ID = up_id + 2
    n_clause = 0

    bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=False,
                                     output_hidden_states=False)
    for param in bert.parameters():
        param.requires_grad = False

    data = pd.read_csv(input_file, sep='\t', encoding='UTF-8', header=0)
    for index, line in data.iterrows():
        n_clause += 1
        senID, clause_no, label, context_w, emotion_w, chatbot_w, words = line
        word_pos = clause_no + 69

        if senID <= up_id:
            continue
        elif senID > bottom_id:
            break

        if next_ID == senID:
            clause_all, tmp_clause_len, relative_pos_all = [], [], []
            next_ID = senID + 1

        if not context_w and emotion_w:
            doc_len.append(len(clause_all))
            relative_pos_all = [array - clause_no for array in relative_pos_all]
            for j in range(max_doc_len - len(clause_all)):
                clause_all.append(np.zeros((max_sen_len, 768)))
                tmp_clause_len.append(0)
                relative_pos_all.append(np.zeros((max_sen_len,)))
            relative_pos.append(relative_pos_all)
            x.append(clause_all)
            sen_len.append(tmp_clause_len)
            clause_all, tmp_clause_len, relative_pos_all = [], [], []
            index_list.append([senID, clause_no, True])
        else:
            index_list.append([senID, clause_no, False])
        encoded_dict = tokenizer.encode_plus(words,
                         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                         max_length=max_sen_len,  # Pad & truncate all sentences.
                         pad_to_max_length=True,
                         return_attention_mask=True,  # Construct attn. masks.
                         return_tensors='pt',  # Return pytorch tensors.
                         truncation_strategy='longest_first')
        clause, _ = bert(encoded_dict['input_ids'], token_type_ids=None, attention_mask=encoded_dict['attention_mask'])
        relative_pos_clause = [word_pos] * max_sen_len
        relative_pos_all.append(np.array(relative_pos_clause))
        clause_all.append(clause.cpu().numpy()[0])
        tmp_clause_len.append(sum(encoded_dict['attention_mask'][0]))
        del encoded_dict
        gc.collect()

    relative_pos, x, sen_len, doc_len = map(np.array, [relative_pos, x, sen_len, doc_len])
    pk.dump(relative_pos, open(path + '{}-test_relative_pos.txt'.format(round+1), 'wb'))
    pk.dump(x, open(path + '{}-test_x_pred.txt'.format(round+1), 'wb'))
    pk.dump(sen_len, open(path + '{}-test_sen_len_pred.txt'.format(round+1), 'wb'))
    pk.dump(doc_len, open(path + '{}-test_doc_len_pred.txt'.format(round+1), 'wb'))
    index_list = np.array(index_list)
    index_list = pd.DataFrame(index_list)
    index_list.to_csv(path + '{}-evaluation_file.csv'.format(round+1), sep='\t', header=None, index=False)
    print('relative_pos.shape {}\nx.shape {} \nsen_len.shape {} \ndoc_len.shape {}\n'.format(
        relative_pos.shape, x.shape, sen_len.shape, doc_len.shape
    ))
    print('test_n_clause {}'.format(n_clause))
    print('load test data done!\n')
    del relative_pos, x, sen_len, doc_len, bert, data
    gc.collect()

def revise_test_data(input_file, predict_path, output_file=None):
    print('load data...')

    pred_prob = list()
    for round in range(5):
        pred_file = predict_path + 'pred_{}_prob.txt'.format(round + 1)
        pred_y = pk.load(open(pred_file, 'rb'))
        pred_prob.extend(pred_y)
    cur_ID = 0
    next_ID = cur_ID + 1
    new_data = list()
    counter = 0
    data = pd.read_csv(input_file, sep='\t', encoding='UTF-8', header=0)
    generator = nextLine(pred_prob)
    for index, line in data.iterrows():
        senID, clause_no, _, context_w, emotion_w, chatbot_w, words = line
        if cur_ID > senID:
            continue
        if senID == next_ID:
            next_ID += 1
        if not context_w and emotion_w:
            try:
                prob = next(generator)
            except StopIteration:
                print(counter)
                break
            counter += 1
            probList = [sen[1] for sen in prob[:clause_no+1]]
        else:
            probList = []
        tmp = [senID, clause_no, context_w, emotion_w, chatbot_w, words, probList]
        new_data.append(tmp)

    col_name = ['conv_id', 'clause_no', 'context?', 'emotion?', 'chatbot?', 'clause', 'prob']
    new_data = pd.DataFrame(new_data)
    new_data.to_csv(output_file, encoding='UTF-8', sep='\t', index=False, header=col_name)

def nextLine(AList):
    for line in AList:
        yield line


# >>>>>>>>>> get some commonsense data >>>>>>>>> #
# process_data(path+'reman-text.csv', path+'reman-ece.csv', path+'clause_keywords.csv', context=False)

# >>>>>>>>>> position information >>>>>>>>> #
# load_w2v(50)

# >>>>>>>>>> load training data >>>>>>>>>> #
# load_data(path+'clause_keywords.csv')

# >>>>>>>>>> load testing data >>>>>>>>>> #total_len = 0
# >>>>>>>>>> There are 19,531 instances >>>>>>>>> #
add_num = 135
up_id = 694
bottom_id = up_id + add_num
sys.stdout = open('./test_dataset_log_3', 'w')
round = 5
while True:
    print('>>>>>>>>>> round {}, up_id: {}, bottom id: {} >>>>>>>>>>'.format(round+1, up_id, bottom_id))
    try:
        load_test_data('./empatheticdialogues/clause_keywords.csv', round, up_id, bottom_id)
        up_id = bottom_id
        add_num += 5
        bottom_id += add_num
        round += 1
    except OverflowError:
        add_num -= 10
        bottom_id -= 10
        print('Add {} per turn'.format(add_num))
    except MemoryError:
        print('Faced MemoryError')
        break
    if up_id > 19531:
        break
    gc.collect()


# >>>>>>>>>> combine the prediction and original data >>>>>>>>>> #
# revise_test_data('./empatheticdialogues/clause_keywords.csv', './empatheticdialogues/', './empatheticdialogues/clause_view-.csv')
