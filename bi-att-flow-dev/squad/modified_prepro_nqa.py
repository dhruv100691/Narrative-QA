import argparse
import json
import os
import pandas as pd
import nltk
from stanfordcorenlp import StanfordCoreNLP
import json
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

from tqdm import tqdm
from squad.utils import get_word_span, get_word_idx, process_tokens


def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    home = os.path.join(home,"Documents","cs546")
    print (home)
    source_dir = os.path.join(home, "data", "narrativeqa-master")
    target_dir = "data/narrativeqa"
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument("--train_name", default='train-v1.1.json')
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--split", action='store_true')
    parser.add_argument("--suffix", default="")
    # TODO : put more args here
    return parser.parse_args()


def create_all(args):
    out_path = os.path.join(args.source_dir, "all-v1.1.json")
    if os.path.exists(out_path):
        return
    train_path = os.path.join(args.source_dir, args.train_name)
    train_data = json.load(open(train_path, 'r'))
    dev_path = os.path.join(args.source_dir, args.dev_name)
    dev_data = json.load(open(dev_path, 'r'))
    train_data['data'].extend(dev_data['data'])
    print("dumping all data ...")
    json.dump(train_data, open(out_path, 'w'))


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    if args.mode == 'full':
        prepro_each(args, 'train', out_name='train')
        prepro_each(args, 'valid', out_name='dev')
        prepro_each(args, 'test', out_name='test')
    elif args.mode == 'all':
        create_all(args)
        prepro_each(args, 'dev', 0.0, 0.0, out_name='dev')
        prepro_each(args, 'dev', 0.0, 0.0, out_name='test')
        prepro_each(args, 'all', out_name='train')
    elif args.mode == 'single':
        assert len(args.single_path) > 0
        prepro_each(args, "NULL", out_name="single", in_path=args.single_path)
    else:
        prepro_each(args, 'train', 0.0, args.train_ratio, out_name='train')
        prepro_each(args, 'train', args.train_ratio, 1.0, out_name='dev')
        prepro_each(args, 'dev', out_name='test')


def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    if args.tokenizer == "PTB":
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        from my.corenlp_interface import CoreNLPInterface
        interface = CoreNLPInterface(args.url, args.port)
        sent_tokenize = interface.split_doc
        word_tokenize = interface.split_sent
    else:
        raise Exception()

    #if not args.split:
    #    sent_tokenize = lambda para: [para]

    source_path = in_path or os.path.join(args.source_dir, "third_party","wikipedia")
    source_summaries =  pd.read_csv(source_path + '/summaries.csv')
    source_qas = pd.read_csv(args.source_dir + '/qaps.csv')

    summaries =[]
    summaries_char_list = []
    ques_answers = []
    questions = []
    questions_char_list =[]
    ques_answer_lengths = []
    document_ids = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    summary_index = -1
    for index_summ, row in tqdm(source_summaries.iterrows(),total=1572):
        if data_type == row['set']:
            summary_tokenized_paras=[]
            summary_char_para = []
            summary_tokenized = list(map(word_tokenize,sent_tokenize(row['summary_tokenized'])))
            summary_tokenized = [process_tokens(tokens) for tokens in summary_tokenized]
            char_list = [[list(word) for word in sent] for sent in summary_tokenized]

            summary_tokenized_paras.append(summary_tokenized) # TODO:each summary has only one paragraph
            summaries.append(summary_tokenized_paras)
            summary_char_para.append(char_list)# TODO:each summary has only one paragraph
            summaries_char_list.append(summary_char_para)
            #coz train/test/valid all are in one file, index_summ cannot be used
            summary_index = summary_index + 1


            qas = source_qas[source_qas['document_id'].isin([row['document_id']])]

            for sent in summary_tokenized:
                for word in sent:
                    word_counter[word] += len(qas)
                    lower_word_counter[word.lower()] += len(qas)
                    for char in word:
                        char_counter[char] += len(qas)

            for index,qa in qas.iterrows() :
                #if question is of multiple sentences, not handling that case also
                #Not req most probably
                question_tokenized = word_tokenize(qa['question'])
                question_tokenized = process_tokens(question_tokenized)
                #print (question_tokenized)
                question_char_list = [list(word) for word in question_tokenized]

                answer1_tokenized = list(map(word_tokenize, sent_tokenize(qa['answer1'])))
                answer1_tokenized = [process_tokens(tokens) for tokens in answer1_tokenized]
                #if answers are of multiple lengths, appending EOS to the last sentence
                answer1_tokenized[len(answer1_tokenized)-1].append('</s>')#appending end token
                answer1_tokenized[0].insert(0,'--SOS--')
                target_length=sum([len(sen) for sen in answer1_tokenized])

                #answer2_tokenized = list(map(word_tokenize, sent_tokenize(qa['answer2'])))
                #answer2_tokenized = [process_tokens(tokens) for tokens in answer2_tokenized]
                #answer2_eos = answer2_tokenized[len(answer2_tokenized) - 1] + ['</s>']  # appending end token
                #answer2_sos = ['--SOS--'] + answer2_tokenized[0]
                #print(answer2_tokenized)

                ques_answers.append(answer1_tokenized)
                ques_answer_lengths.append(target_length)
                questions.append(question_tokenized)
                questions_char_list.append(question_char_list)
                document_ids.append([summary_index,row['document_id']])

                for sent in question_tokenized:
                    for word in sent:
                        word_counter[word] += 1
                        lower_word_counter[word.lower()] += 1
                        for char in word:
                            char_counter[char] += 1

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    data = {'q': questions, 'cq': questions_char_list, '*x': document_ids,
            'answerss': ques_answers , '*cx': document_ids ,'ans_len': ques_answer_lengths}
    shared = {'x': summaries, 'cx': summaries_char_list,'word_counter': word_counter,
              'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

    print("saving ...")
    save(args, data, shared, out_name)



if __name__ == "__main__":
    main()
