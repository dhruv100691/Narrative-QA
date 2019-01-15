'''
A sample code usage of the python package stanfordcorenlp to access a Stanford CoreNLP server.
Written as part of the blog post: https://www.khalidalnajjar.com/how-to-setup-and-use-stanford-corenlp-server-with-python/
'''

from stanfordcorenlp import StanfordCoreNLP
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from random import choice
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'ner,dcoref',
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

    def preprocess_summary(self,text,index):
        results = self.annotate(text)
        index_set=set()
        entity_id=0
        summary = []
        summary_wo_categories = []
        entity_dict = defaultdict(list)
        entity_dict_wo = defaultdict(list)
        for sent in results['sentences']: #doing this coz, could not find sentence tokenizer in the Stanford library
            start_offset = sent['tokens'][0]['characterOffsetBegin']  # Begin offset of first token.
            end_offset = sent['tokens'][-1]['characterOffsetEnd']  # End offset of last token.
            sent_str = text[start_offset:end_offset]
            summary.append(sNLP.word_tokenize(sent_str))
            summary_wo_categories.append(sNLP.word_tokenize(sent_str))

        for id, entities in enumerate(results['corefs'].values()):
            type_tag='_O_'
            if entities[0]['type'] in {'PROPER'}:
                r_entity = [e for e in entities if e['isRepresentativeMention']]
                type_tag = results['sentences'][r_entity[0]['sentNum'] - 1]['tokens'][r_entity[0]['startIndex'] - 1]['ner']
                self.tagset.add(type_tag)
                type_tag = '_' + type_tag + '_'
                entity_id = choice([i for i in range(1,max(len(index_set)+5,20)) if i not in index_set])

            for entity in entities:
                if entity['type'] in {'PROPER'}:
                    sentence = summary[entity['sentNum'] - 1]
                    sentence_wo_category = summary_wo_categories[entity['sentNum'] - 1]
                    if any('@entity_' in x for x in sentence[entity['startIndex'] - 1 : entity['endIndex']]):
                        continue
                    if any('NULL' in x for x in sentence[entity['startIndex'] - 1 : entity['endIndex']]):
                        continue
                    sentence[entity['startIndex'] - 1] = " @entity" + type_tag +str(entity_id) + " "
                    sentence_wo_category[entity['startIndex'] - 1] = " @entity_" + str(entity_id) +" "
                    self.idset.add(entity_id)
                    index_set.add(entity_id)
                    entity_dict["entity" + type_tag +str(entity_id)].append(entity)
                    entity_dict_wo["entity_" +str(entity_id)].append(entity)
                    for i in range(entity['startIndex'], entity['endIndex'] - 1):
                        sentence[i] = 'NULL'
                        sentence_wo_category[i] = 'NULL'
                    summary[entity['sentNum'] - 1] = sentence
                    summary_wo_categories[entity['sentNum'] - 1] = sentence_wo_category

        for sent,sent_wo_category in zip(summary,summary_wo_categories):
            sent[:] = [x for x in sent if x != 'NULL']
            sent_wo_category[:] = [x for x in sent_wo_category if x != 'NULL']

        return (summary,summary_wo_categories,entity_dict,entity_dict_wo)

if __name__ == '__main__':
    sNLP = StanfordNLP()
    #print("POS:", sNLP.pos(text))
    # print ("Annotate:",results['corefs'])
    # print ("Tokens:", sNLP.word_tokenize(text))
    # print ("NER:", sNLP.ner(text))
    # print ("Parse:", sNLP.parse(text))
    # print ("Dep Parse:", sNLP.dependency_parse(text))

    source_summaries = pd.read_csv('/Users/dhruv100691/Documents/cs546/data/narrativeqa-master/third_party/wikipedia/summaries.csv')
    source_qas = pd.read_csv('/Users/dhruv100691/Documents/cs546/data/narrativeqa-master/qaps.csv')
    #assign empty columns
    source_qas =source_qas.assign(processed_question='',processed_answer='',processed_question_wo='',processed_answer_wo='')
    source_summaries=source_summaries.assign(processed_summary='',processed_summary_wo='')
    final_qas = pd.DataFrame()
    entity_dict = dict()
    entity_dict_wo = dict()

    for index_summ, row in tqdm(source_summaries.iterrows(),total=1572):
        summary = row['summary_tokenized']
        qas= source_qas[source_qas['document_id'].isin([row['document_id']])]
        qas=qas.reset_index(drop=True)
        for qid,ques_row in qas.iterrows():
            sent = sNLP.word_tokenize(ques_row['answer1_tokenized'])
            if sent[-1] != '.':
                sent.append('.')
                ques_row['answer1_tokenized']= ' '.join(sent)
            summary = summary + '\nquestion ' + ques_row['question_tokenized'] + '\n'+ques_row['answer1_tokenized']
        summary,summary_wo_cat,entity_dict[row['document_id']],entity_dict_wo[row['document_id']] = sNLP.preprocess_summary(summary,index_summ)
        summary_processed = ''
        summary_wo_cat_processed=''
        qid=-1
        for i in range(len(summary)):
            sent=summary[i]
            sent_wo=summary_wo_cat[i]
            if sent[0]=='question':
                break
            summary_processed += ' '.join(sent)
            summary_processed += ' '
            summary_wo_cat_processed += ' '.join(sent_wo)
            summary_wo_cat_processed += ' '

        for j in range (i,len(summary)):
            sent=summary[j]
            sent_wo = summary_wo_cat[j]
            if sent[0]=='question':
                qid+=1
                qas.loc[qid,'processed_question']= ' '.join(sent[1:])
                qas.loc[qid, 'processed_question_wo'] = ' '.join(sent_wo[1:])
            else:
                qas.loc[qid,'processed_answer'] = qas.loc[qid,'processed_answer'] + ' '.join(sent)
                qas.loc[qid, 'processed_answer_wo'] = qas.loc[qid, 'processed_answer_wo'] + ' '.join(sent_wo)
        source_summaries.loc[index_summ,'processed_summary']=summary_processed
        source_summaries.loc[index_summ,'processed_summary_wo']=summary_wo_cat_processed
        final_qas=final_qas.append(qas)
        print (index_summ)
        if (index_summ+1) % 3 ==0:
            break

    final_qas.to_csv('processed_qas_new_try.csv')
    source_summaries.to_csv('processed_summaries_new_try.csv')
    json.dump(entity_dict, open('entity_dict_try.json', 'w'))
    json.dump(entity_dict_wo, open('entity_dict_wo_try.json', 'w'))
    print(sNLP.tagset)
    print(sNLP.idset)



