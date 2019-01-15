import pandas as pd
import nltk
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
import time
from squad.utils import process_tokens
from rouge import Rouge

source_summaries = pd.read_csv('processed_summaries_new.csv')
source_qas = pd.read_csv('processed_answer_spans_rouge_wo_preprocess.csv')
source_qas['start_index'] = source_qas['start_index'].str.replace('(','').str.replace(')','')
source_qas['end_index'] = source_qas['end_index'].str.replace('(','').str.replace(')','')
source_qas.dropna(subset=['start_index','end_index'],inplace=True)
#assign empty columns
#source_qas =source_qas.assign(start_index='',end_index='')
final_qas = pd.DataFrame()

def modify_answer_spans(qas,summary):
    #converts span from (sent_num, word_num) -> (0,total_word_num in para)
    import nltk
    summary_tokenized = list(map(nltk.word_tokenize, nltk.sent_tokenize(summary)))
    summary_tokenized = [process_tokens(tokens) for tokens in summary_tokenized]

    for index, qa in qas.iterrows():
        answer1_span_start_idx = qa['start_index'].split(', ')
        answer1_span_end_idx = qa['end_index'].split(', ')
        answer1_span_start_idx = list(map(int, answer1_span_start_idx))
        answer1_span_end_idx = list(map(int, answer1_span_end_idx))
        index_mod = sum(map(len,summary_tokenized[0:answer1_span_start_idx[0]]))
        qas.at[index,'start_index'] = [0,index_mod + answer1_span_start_idx[1]]
        index_mod = sum(map(len, summary_tokenized[0:answer1_span_end_idx[0]]))
        qas.at[index,'end_index'] = [0, index_mod + answer1_span_end_idx[1]]
    return qas

def evaluate_bleu_scores(data_type):
    bleu_scores=[]
    bleu_4_scores=[]
    for index_summ, row in tqdm(source_summaries.iterrows(),total=1572):
        if data_type == row['set']:
            references=[]
            references1=[]
            spans=[]
            #summary = row['processed_summary'].replace(".",". ")
            #summ = list(map(nltk.word_tokenize, nltk.sent_tokenize(row['processed_summary_wo'])))
            #summ = [process_tokens(tokens) for tokens in summ]
            summary_tokenized = nltk.word_tokenize(row['summary_tokenized'])
            summary_tokenized = list(map(str.lower,process_tokens(summary_tokenized)))
            qas= source_qas[source_qas['document_id'].isin([row['document_id']])]
            qas=qas.reset_index(drop=True)
            qas = modify_answer_spans(qas, row['summary_tokenized'])
            for qid,ques_row in qas.iterrows():
                sent = list(map(str.lower,nltk.word_tokenize(ques_row['answer1_tokenized'].replace(".",""))))
                #print ("Question",qid,ques_row['processed_question_wo'])
                #print ("Answer:",sent)
                #print("indices",ques_row['start_index'],ques_row['end_index'])
                predicted_rouge_span= summary_tokenized[ques_row['start_index'][1]:ques_row['end_index'][1]+1]
                #print ("Rouge Span:",predicted_rouge_span)
                references.append([sent])
                #references1.append([predicted_rouge_span])
                spans.append(predicted_rouge_span)
            bleu_scores.append(corpus_bleu(references,spans,weights=(1,0,0,0)))
            bleu_4_scores.append(corpus_bleu(references,spans))
    print ("Average score bleu_1 for",data_type,sum(bleu_scores)/len(bleu_scores))
    print ("Average score bleu_4 for",data_type,sum(bleu_4_scores)/len(bleu_4_scores))

evaluate_bleu_scores('train')
evaluate_bleu_scores('valid')
evaluate_bleu_scores('test')
