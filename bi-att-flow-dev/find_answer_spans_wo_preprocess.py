import pandas as pd
import nltk
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import time
from squad.utils import process_tokens
from rouge import Rouge

source_summaries = pd.read_csv('processed_summaries_new.csv')
source_qas = pd.read_csv('processed_qas_new.csv')
source_qas.dropna(inplace=True)
#assign empty columns
source_qas =source_qas.assign(start_index='',end_index='')
final_qas = pd.DataFrame()
rouge= Rouge()

def get_all_substrings(input_string):
  length = len(input_string)
  #yield [input_string[i:j+1] for i in range(length) for j in range(i,length)]
  for i in range(length):
      for j in range(i, length):
          yield input_string[i:j+1]

def get_2d_span(summary,index):
    sum=0
    for sent_num,sent in enumerate(summary):
        if index <(sum + len(sent)):
            return (sent_num,index-sum)
        sum+=len(sent)

for index_summ, row in tqdm(source_summaries.iterrows(),total=1572):
    summ = list(map(nltk.word_tokenize,nltk.sent_tokenize(row['summary_tokenized'])))
    summ = [process_tokens(tokens) for tokens in summ]
    summary_tokenized = nltk.word_tokenize(row['summary_tokenized'])
    summary_tokenized = list(map(str.lower,process_tokens(summary_tokenized)))
    qas= source_qas[source_qas['document_id'].isin([row['document_id']])]
    qas=qas.reset_index(drop=True)
    list_ex = [".", "..", "..."]
    for qid,ques_row in qas.iterrows():
        sent = list(map(str.lower,nltk.word_tokenize(ques_row['answer1_tokenized'].replace(".",""))))
        max =0
        ans_span=[]
        actual_ans_str = ' '.join(sent)
        for sub_str in get_all_substrings(summary_tokenized):
            if abs(len(sub_str) - len(sent)) < 10:
                #score = sentence_bleu([sub_str], sent, weights=(1, 0, 0, 0))
                ref_str = ' '.join(sub_str)
                if ref_str not in list_ex and len(ref_str) > 0:
                    try:
                        scores = rouge.get_scores(actual_ans_str, ref_str)
                    except:
                        print("actual_str", actual_ans_str)
                        print("REFSTR", ref_str)
                    if scores[0]["rouge-l"]["f"] > max:
                        max = scores[0]["rouge-l"]["f"]
                        ans_span = sub_str
        if ans_span == []:
            continue
        index_list = [i for i, word in enumerate(summary_tokenized) if word == ans_span[0]]
        for start_idx in index_list:
            if summary_tokenized[start_idx:(start_idx + len(ans_span))] == ans_span:
                qas.at[qid,'start_index']  = get_2d_span(summ, start_idx)
                qas.at[qid, 'end_index'] = get_2d_span(summ, (start_idx + len(ans_span) - 1))
                break
    final_qas = final_qas.append(qas)

final_qas.to_csv('processed_answer_spans_rouge_wo_preprocess.csv')
