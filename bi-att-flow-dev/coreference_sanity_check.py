'''
A sample code usage of the python package stanfordcorenlp to access a Stanford CoreNLP server.
Written as part of the blog post: https://www.khalidalnajjar.com/how-to-setup-and-use-stanford-corenlp-server-with-python/
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000
'''

from stanfordcorenlp import StanfordCoreNLP
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from random import choice
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
from itertools import groupby

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'ner dcoref',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }
        self.tagset=set()
        self.idset = set()

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

def run_coref(words,entity_dict,index_set):
    key = ' '.join(words[0])
    if key in entity_dict:
        words[0] = [entity_dict[key]]
    else:
        for k in entity_dict.keys():
            if k.find(key) != -1:
                words[0] = [entity_dict[k]]
                return
        entity_id = choice([i for i in range(1, max(len(index_set) + 5, 20)) if i not in index_set])
        entity_dict[key] = 'entity_' + str(entity_id)
        words[0] = [entity_dict[key]]
        index_set.add(entity_id)

def preprocess_data():
    sNLP = StanfordNLP()
    # print("POS:", sNLP.pos(text))
    # print ("Annotate:",results['corefs'])
    # print ("Tokens:", sNLP.word_tokenize(text))
    # print ("NER:", sNLP.ner(text))
    # print ("Parse:", sNLP.parse(text))
    # print ("Dep Parse:", sNLP.dependency_parse(text))

    source_summaries = pd.read_csv(
        '/Users/dhruv100691/Documents/cs546/data/narrativeqa-master/third_party/wikipedia/summaries.csv')
    source_qas = pd.read_csv('/Users/dhruv100691/Documents/cs546/data/narrativeqa-master/qaps.csv')
    # assign empty columns
    source_qas = source_qas.assign(processed_question='', processed_answer='')
    source_summaries = source_summaries.assign(processed_summary='')
    final_qas = pd.DataFrame()
    entity_dict_wo = dict()
    bleu_scores = []

    for index_summ, row in tqdm(source_summaries.iterrows(), total=1572):
        index_set = set()
        entity_dict = dict()
        summary = row['summary_tokenized']
        qas = source_qas[source_qas['document_id'].isin([row['document_id']])]
        qas = qas.reset_index(drop=True)
        output = sNLP.ner(summary)

        modified_output = []
        for tag, chunk in groupby(output, lambda x: x[1]):
            modified_output.append([[w for w, t in chunk], tag])
        for idx, words in enumerate(modified_output):
            if words[1] in ['PERSON', 'ORGANIZATION', 'STATE_OR_PROVINCE', 'CITY']:
                run_coref(words,entity_dict,index_set)
        processed_summary = ''
        for words in modified_output:
            if len(words[0]) > 1:
                processed_summary += ' '.join(words[0])
            else:
                processed_summary += words[0][0]
            processed_summary += ' '
        source_summaries.loc[index_summ, 'processed_summary'] = processed_summary

        for qid, ques_row in qas.iterrows():
            out_ans = sNLP.ner(ques_row['answer1_tokenized'])
            out_ques = sNLP.ner(ques_row['question_tokenized'])
            ans_ner=[]
            ques_ner=[]
            for tag, chunk in groupby(out_ans, lambda x: x[1]):
                ans_ner.append([[w for w, t in chunk], tag])
            for tag, chunk in groupby(out_ques, lambda x: x[1]):
                ques_ner.append([[w for w, t in chunk], tag])
            for idx, words in enumerate(ans_ner):
                if words[1] in ['PERSON', 'ORGANIZATION', 'STATE_OR_PROVINCE', 'CITY']:
                    run_coref(words, entity_dict, index_set)
            processed_str=''
            for words in ans_ner:
                if len(words[0]) >1:
                    processed_str += ' '.join(words[0])
                else:
                    processed_str += words[0][0]
                processed_str+=' '
            qas.loc[qid, 'processed_answer'] = processed_str

            for idx, words in enumerate(ques_ner):
                if words[1] in ['PERSON', 'ORGANIZATION', 'STATE_OR_PROVINCE', 'CITY']:
                    run_coref(words, entity_dict, index_set)
            processed_str=''
            for words in ques_ner:
                if len(words[0]) >1:
                    processed_str += ' '.join(words[0])
                else:
                    processed_str += words[0][0]
                processed_str+=' '

            qas.loc[qid, 'processed_question'] = processed_str

        final_qas=final_qas.append(qas)
        entity_dict_wo[row['document_id']] = entity_dict

        '''
        for qid,ques_row in qas.iterrows():
            sent = sNLP.word_tokenize(ques_row['answer1_tokenized'])
            if sent[-1] != '.':
                sent.append('.')
                ques_row['answer1_tokenized']= ' '.join(sent)
            summary = summary + '\nquestion ' + ques_row['question_tokenized'] + '\n'+ques_row['answer1_tokenized']
        summary,summary_wo_cat,entity_dict[row['document_id']],entity_dict_wo[row['document_id']],original_summ = sNLP.preprocess_summary(summary,index_summ)
        #print ("Summary",summary_wo_cat)
        #summary_wo_cat_unmodified=[[sent] for sent in original_summ]
        summary_wo_cat_unmodified = [[[item for sublist in original_summ for item in sublist]]]
        count=0
        for k,v in entity_dict_wo.items(): #for every doc
            for entities,corefs in v.items(): #for every entity
                #print ("ENTITY",entities)
                for ref in corefs:
                    #print ("Position",ref)
                    #print ("SENT",summary_wo_cat[ref['position'][0] -1])
                    try:
                        summary_wo_cat[ref['position'][0] - 1][ref['startIndex'] - 1]= ref['text']
                    except:
                        count+=1
                    #print ("Modified Sent",summary_wo_cat[ref['position'][0] -1])
        #print("replace failed",count)
        summary_wo_cat = [[item for sublist in summary_wo_cat for item in sublist]]
        #print ("Summary NEW",summary_wo_cat)
        #print ("Summary unmodified",summary_wo_cat_unmodified)
        bleu_scores.append(corpus_bleu(summary_wo_cat_unmodified,summary_wo_cat))
        print ("Scores, errors",bleu_scores[-1],count)
        '''

    final_qas.to_csv('processed_qas_new_method.csv')
    source_summaries.to_csv('processed_summaries_new_method.csv')
    json.dump(entity_dict_wo, open('entity_dict_new_method.json', 'w'))


if __name__ == '__main__':
    preprocess_data()

