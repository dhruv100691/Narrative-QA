import numpy as np
import tensorflow as tf
from basic.read_data import DataSet
from my.nltk_utils import span_f1
from my.tensorflow import padded_reshape
from my.utils import argmax
from squad.utils import get_phrase, get_best_span, get_best_span_wy
import collections
import math
from nltk.translate.bleu_score import corpus_bleu

class Evaluation(object):
    def __init__(self, data_type, global_step, idxs, yp, tensor_dict=None):
        self.data_type = data_type
        self.global_step = global_step
        self.idxs = idxs
        self.yp = yp
        self.num_examples = len(yp)
        self.tensor_dict = None
        self.dict = {'data_type': data_type,
                     'global_step': global_step,
                     'yp': yp,
                     'idxs': idxs,
                     'num_examples': self.num_examples}
        if tensor_dict is not None:
            self.tensor_dict = {key: val.tolist() for key, val in tensor_dict.items()}
            for key, val in self.tensor_dict.items():
                self.dict[key] = val
        self.summaries = None

    def __repr__(self):
        return "{} step {}".format(self.data_type, self.global_step)

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_yp = self.yp + other.yp
        new_idxs = self.idxs + other.idxs
        new_tensor_dict = None
        if self.tensor_dict is not None:
            new_tensor_dict = {key: val + other.tensor_dict[key] for key, val in self.tensor_dict.items()}
        return Evaluation(self.data_type, self.global_step, new_idxs, new_yp, tensor_dict=new_tensor_dict)

    def __radd__(self, other):
        return self.__add__(other)


class LabeledEvaluation(Evaluation):
    def __init__(self, data_type, global_step, idxs, yp, y, tensor_dict=None):
        super(LabeledEvaluation, self).__init__(data_type, global_step, idxs, yp, tensor_dict=tensor_dict)
        self.y = y
        self.dict['y'] = y

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_yp = self.yp + other.yp
        new_y = self.y + other.y
        new_idxs = self.idxs + other.idxs
        if self.tensor_dict is not None:
            new_tensor_dict = {key: np.concatenate((val, other.tensor_dict[key]), axis=0) for key, val in self.tensor_dict.items()}
        return LabeledEvaluation(self.data_type, self.global_step, new_idxs, new_yp, new_y, tensor_dict=new_tensor_dict)


class AccuracyEvaluation(LabeledEvaluation):
    def __init__(self, data_type, global_step, idxs, yp, y, correct, loss, tensor_dict=None):
        super(AccuracyEvaluation, self).__init__(data_type, global_step, idxs, yp, y, tensor_dict=tensor_dict)
        self.loss = loss
        self.correct = correct
        self.acc = sum(correct) / len(correct)
        self.dict['loss'] = loss
        self.dict['correct'] = correct
        self.dict['acc'] = self.acc
        loss_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/loss'.format(data_type), simple_value=self.loss)])
        acc_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/acc'.format(data_type), simple_value=self.acc)])
        self.summaries = [loss_summary, acc_summary]

    def __repr__(self):
        return "{} step {}: accuracy={}, loss={}".format(self.data_type, self.global_step, self.acc, self.loss)

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_idxs = self.idxs + other.idxs
        new_yp = self.yp + other.yp
        new_y = self.y + other.y
        new_correct = self.correct + other.correct
        new_loss = (self.loss * self.num_examples + other.loss * other.num_examples) / len(new_correct)
        if self.tensor_dict is not None:
            new_tensor_dict = {key: np.concatenate((val, other.tensor_dict[key]), axis=0) for key, val in self.tensor_dict.items()}
        return AccuracyEvaluation(self.data_type, self.global_step, new_idxs, new_yp, new_y, new_correct, new_loss, tensor_dict=new_tensor_dict)


class Evaluator(object):
    def __init__(self, config, model, tensor_dict=None):
        self.config = config
        self.model = model
        self.global_step = model.global_step
        #self.yp = model.yp
        self.tensor_dict = {} if tensor_dict is None else tensor_dict

    def get_evaluation(self, sess, batch):
        idxs, data_set = batch
        feed_dict = self.model.get_feed_dict(data_set, False, supervised=False)
        global_step, yp, vals = sess.run([self.global_step, self.yp, list(self.tensor_dict.values())], feed_dict=feed_dict)
        yp = yp[:data_set.num_examples]
        tensor_dict = dict(zip(self.tensor_dict.keys(), vals))
        e = Evaluation(data_set.data_type, int(global_step), idxs, yp.tolist(), tensor_dict=tensor_dict)
        return e

    def get_evaluation_from_batches(self, sess, batches):
        e = sum(self.get_evaluation(sess, batch) for batch in batches)
        return e


class LabeledEvaluator(Evaluator):
    def __init__(self, config, model, tensor_dict=None):
        super(LabeledEvaluator, self).__init__(config, model, tensor_dict=tensor_dict)
        self.y = model.y

    def get_evaluation(self, sess, batch):
        idxs, data_set = batch
        feed_dict = self.model.get_feed_dict(data_set, False, supervised=False)
        global_step, yp, vals = sess.run([self.global_step, self.yp, list(self.tensor_dict.values())], feed_dict=feed_dict)
        yp = yp[:data_set.num_examples]
        y = feed_dict[self.y]
        tensor_dict = dict(zip(self.tensor_dict.keys(), vals))
        e = LabeledEvaluation(data_set.data_type, int(global_step), idxs, yp.tolist(), y.tolist(), tensor_dict=tensor_dict)
        return e


class AccuracyEvaluator(LabeledEvaluator):
    def __init__(self, config, model, tensor_dict=None):
        super(AccuracyEvaluator, self).__init__(config, model, tensor_dict=tensor_dict)
        self.loss = model.loss

    def get_evaluation(self, sess, batch):
        idxs, data_set = batch
        assert isinstance(data_set, DataSet)
        feed_dict = self.model.get_feed_dict(data_set, False)
        global_step, yp, loss, vals = sess.run([self.global_step, self.yp, self.loss, list(self.tensor_dict.values())], feed_dict=feed_dict)
        y = data_set.data['y']
        yp = yp[:data_set.num_examples]
        correct = [self.__class__.compare(yi, ypi) for yi, ypi in zip(y, yp)]
        tensor_dict = dict(zip(self.tensor_dict.keys(), vals))
        e = AccuracyEvaluation(data_set.data_type, int(global_step), idxs, yp.tolist(), y, correct, float(loss), tensor_dict=tensor_dict)
        return e

    @staticmethod
    def compare(yi, ypi):
        for start, stop in yi:
            if start == int(np.argmax(ypi)):
                return True
        return False


class AccuracyEvaluator2(AccuracyEvaluator):
    @staticmethod
    def compare(yi, ypi):
        for start, stop in yi:
            para_start = int(np.argmax(np.max(ypi, 1)))
            sent_start = int(np.argmax(ypi[para_start]))
            if tuple(start) == (para_start, sent_start):
                return True
        return False


class ForwardEvaluation(Evaluation):
    def __init__(self, data_type, global_step, idxs, yp, yp2, loss, id2answer_dict, tensor_dict=None):
        super(ForwardEvaluation, self).__init__(data_type, global_step, idxs, yp, tensor_dict=tensor_dict)
        self.yp2 = yp2
        self.loss = loss
        self.dict['loss'] = loss
        self.dict['yp2'] = yp2
        self.id2answer_dict = id2answer_dict

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_idxs = self.idxs + other.idxs
        new_yp = self.yp + other.yp
        new_yp2 = self.yp2 + other.yp2
        new_loss = (self.loss * self.num_examples + other.loss * other.num_examples) / len(new_yp)
        new_id2answer_dict = dict(list(self.id2answer_dict.items()) + list(other.id2answer_dict.items()))
        new_id2score_dict = dict(list(self.id2answer_dict['scores'].items()) + list(other.id2answer_dict['scores'].items()))
        new_id2answer_dict['scores'] = new_id2score_dict
        if self.tensor_dict is not None:
            new_tensor_dict = {key: np.concatenate((val, other.tensor_dict[key]), axis=0) for key, val in self.tensor_dict.items()}
        return ForwardEvaluation(self.data_type, self.global_step, new_idxs, new_yp, new_yp2, new_loss, new_id2answer_dict, tensor_dict=new_tensor_dict)

    def __repr__(self):
        return "{} step {}: loss={:.4f}".format(self.data_type, self.global_step, self.loss)


class F1Evaluation(AccuracyEvaluation):
    def __init__(self, data_type, global_step, idxs, yp, yp2, y, correct, loss, f1s, id2answer_dict, tensor_dict=None):
        super(F1Evaluation, self).__init__(data_type, global_step, idxs, yp, y, correct, loss, tensor_dict=tensor_dict)
        self.yp2 = yp2
        self.f1s = f1s
        self.f1 = float(np.mean(f1s))
        self.dict['yp2'] = yp2
        self.dict['f1s'] = f1s
        self.dict['f1'] = self.f1
        self.id2answer_dict = id2answer_dict
        f1_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/f1'.format(data_type), simple_value=self.f1)])
        self.summaries.append(f1_summary)

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_idxs = self.idxs + other.idxs
        new_yp = self.yp + other.yp
        new_yp2 = self.yp2 + other.yp2
        new_y = self.y + other.y
        new_correct = self.correct + other.correct
        new_f1s = self.f1s + other.f1s
        new_loss = (self.loss * self.num_examples + other.loss * other.num_examples) / len(new_correct)
        new_id2answer_dict = dict(list(self.id2answer_dict.items()) + list(other.id2answer_dict.items()))
        new_id2score_dict = dict(list(self.id2answer_dict['scores'].items()) + list(other.id2answer_dict['scores'].items()))
        new_id2answer_dict['scores'] = new_id2score_dict
        if 'na' in self.id2answer_dict:
            new_id2na_dict = dict(list(self.id2answer_dict['na'].items()) + list(other.id2answer_dict['na'].items()))
            new_id2answer_dict['na'] = new_id2na_dict
        e = F1Evaluation(self.data_type, self.global_step, new_idxs, new_yp, new_yp2, new_y, new_correct, new_loss, new_f1s, new_id2answer_dict)
        if 'wyp' in self.dict:
            new_wyp = self.dict['wyp'] + other.dict['wyp']
            e.dict['wyp'] = new_wyp
        return e

    def __repr__(self):
        return "{} step {}: accuracy={:.4f}, f1={:.4f}, loss={:.4f}".format(self.data_type, self.global_step, self.acc, self.f1, self.loss)


class F1Evaluator(LabeledEvaluator):
    def __init__(self, config, model, tensor_dict=None):
        super(F1Evaluator, self).__init__(config, model, tensor_dict=tensor_dict)
        self.yp2 = model.yp2
        self.wyp = model.wyp
        self.loss = model.loss
        if config.na:
            self.na = model.na_prob

    def get_evaluation(self, sess, batch):
        idxs, data_set = self._split_batch(batch)
        assert isinstance(data_set, DataSet)
        feed_dict = self._get_feed_dict(batch)
        if self.config.na:
            global_step, yp, yp2, wyp, loss, na, vals = sess.run([self.global_step, self.yp, self.yp2, self.wyp, self.loss, self.na, list(self.tensor_dict.values())], feed_dict=feed_dict)
        else:
            global_step, yp, yp2, wyp, loss, vals = sess.run([self.global_step, self.yp, self.yp2, self.wyp, self.loss, list(self.tensor_dict.values())], feed_dict=feed_dict)
        y = data_set.data['y']
        if self.config.squash:
            new_y = []
            for xi, yi in zip(data_set.data['x'], y):
                new_yi = []
                for start, stop in yi:
                    start_offset = sum(map(len, xi[:start[0]]))
                    stop_offset = sum(map(len, xi[:stop[0]]))
                    new_start = 0, start_offset + start[1]
                    new_stop = 0, stop_offset + stop[1]
                    new_yi.append((new_start, new_stop))
                new_y.append(new_yi)
            y = new_y
        if self.config.single:
            new_y = []
            for yi in y:
                new_yi = []
                for start, stop in yi:
                    new_start = 0, start[1]
                    new_stop = 0, stop[1]
                    new_yi.append((new_start, new_stop))
                new_y.append(new_yi)
            y = new_y

        yp, yp2, wyp = yp[:data_set.num_examples], yp2[:data_set.num_examples], wyp[:data_set.num_examples]
        if self.config.wy:
            spans, scores = zip(*[get_best_span_wy(wypi, self.config.th) for wypi in wyp])
        else:
            spans, scores = zip(*[get_best_span(ypi, yp2i) for ypi, yp2i in zip(yp, yp2)])

        def _get(xi, span):
            if len(xi) <= span[0][0]:
                return [""]
            if len(xi[span[0][0]]) <= span[1][1]:
                return [""]
            return xi[span[0][0]][span[0][1]:span[1][1]]

        def _get2(context, xi, span):
            if len(xi) <= span[0][0]:
                return ""
            if len(xi[span[0][0]]) <= span[1][1]:
                return ""
            return get_phrase(context, xi, span)

        id2answer_dict = {id_: _get2(context, xi, span)
                          for id_, xi, span, context in zip(data_set.data['ids'], data_set.data['x'], spans, data_set.data['p'])}
        id2score_dict = {id_: score for id_, score in zip(data_set.data['ids'], scores)}
        id2answer_dict['scores'] = id2score_dict
        if self.config.na:
            id2na_dict = {id_: float(each) for id_, each in zip(data_set.data['ids'], na)}
            id2answer_dict['na'] = id2na_dict
        correct = [self.__class__.compare2(yi, span) for yi, span in zip(y, spans)]
        f1s = [self.__class__.span_f1(yi, span) for yi, span in zip(y, spans)]
        tensor_dict = dict(zip(self.tensor_dict.keys(), vals))
        e = F1Evaluation(data_set.data_type, int(global_step), idxs, yp.tolist(), yp2.tolist(), y,
                         correct, float(loss), f1s, id2answer_dict, tensor_dict=tensor_dict)
        if self.config.wy:
            e.dict['wyp'] = wyp.tolist()
        return e

    def _split_batch(self, batch):
        return batch

    def _get_feed_dict(self, batch):
        return self.model.get_feed_dict(batch[1], False)

    @staticmethod
    def compare(yi, ypi, yp2i):
        for start, stop in yi:
            aypi = argmax(ypi)
            mask = np.zeros(yp2i.shape)
            mask[aypi[0], aypi[1]:] = np.ones([yp2i.shape[1] - aypi[1]])
            if tuple(start) == aypi and (stop[0], stop[1]-1) == argmax(yp2i * mask):
                return True
        return False

    @staticmethod
    def compare2(yi, span):
        for start, stop in yi:
            if tuple(start) == span[0] and tuple(stop) == span[1]:
                return True
        return False

    @staticmethod
    def span_f1(yi, span):
        max_f1 = 0
        for start, stop in yi:
            if start[0] == span[0][0]:
                true_span = start[1], stop[1]
                pred_span = span[0][1], span[1][1]
                f1 = span_f1(true_span, pred_span)
                max_f1 = max(f1, max_f1)
        return max_f1


class MultiGPUF1Evaluator(F1Evaluator):
    def __init__(self, config, models, tensor_dict=None):
        super(MultiGPUF1Evaluator, self).__init__(config, models[0], tensor_dict=tensor_dict)
        self.models = models
        with tf.name_scope("eval_concat"):
            N, M, JX = config.batch_size, config.max_num_sents, config.max_sent_size
            self.yp = tf.concat(axis=0, values=[padded_reshape(model.yp, [N, M, JX]) for model in models])
            self.yp2 = tf.concat(axis=0, values=[padded_reshape(model.yp2, [N, M, JX]) for model in models])
            self.wy = tf.concat(axis=0, values=[padded_reshape(model.wy, [N, M, JX]) for model in models])
            self.loss = tf.add_n([model.loss for model in models])/len(models)

    def _split_batch(self, batches):
        idxs_list, data_sets = zip(*batches)
        idxs = sum(idxs_list, ())
        data_set = sum(data_sets, data_sets[0].get_empty())
        return idxs, data_set

    def _get_feed_dict(self, batches):
        feed_dict = {}
        for model, (_, data_set) in zip(self.models, batches):
            feed_dict.update(model.get_feed_dict(data_set, False))
        return feed_dict


class ForwardEvaluator(Evaluator):
    def __init__(self, config, model, tensor_dict=None):
        super(ForwardEvaluator, self).__init__(config, model, tensor_dict=tensor_dict)
        self.yp2 = model.yp2
        self.loss = model.loss
        if config.na:
            self.na = model.na_prob

    def get_evaluation(self, sess, batch):
        idxs, data_set = batch
        assert isinstance(data_set, DataSet)
        feed_dict = self.model.get_feed_dict(data_set, False)
        if self.config.na:
            global_step, yp, yp2, loss, na, vals = sess.run([self.global_step, self.yp, self.yp2, self.loss, self.na, list(self.tensor_dict.values())], feed_dict=feed_dict)
        else:
            global_step, yp, yp2, loss, vals = sess.run([self.global_step, self.yp, self.yp2, self.loss, list(self.tensor_dict.values())], feed_dict=feed_dict)

        yp, yp2 = yp[:data_set.num_examples], yp2[:data_set.num_examples]
        spans, scores = zip(*[get_best_span(ypi, yp2i) for ypi, yp2i in zip(yp, yp2)])

        def _get(xi, span):
            if len(xi) <= span[0][0]:
                return [""]
            if len(xi[span[0][0]]) <= span[1][1]:
                return [""]
            return xi[span[0][0]][span[0][1]:span[1][1]]

        def _get2(context, xi, span):
            if len(xi) <= span[0][0]:
                return ""
            if len(xi[span[0][0]]) <= span[1][1]:
                return ""
            return get_phrase(context, xi, span)

        id2answer_dict = {id_: _get2(context, xi, span)
                          for id_, xi, span, context in zip(data_set.data['ids'], data_set.data['x'], spans, data_set.data['p'])}
        id2score_dict = {id_: score for id_, score in zip(data_set.data['ids'], scores)}
        id2answer_dict['scores'] = id2score_dict
        if self.config.na:
            id2na_dict = {id_: float(each) for id_, each in zip(data_set.data['ids'], na)}
            id2answer_dict['na'] = id2na_dict
        tensor_dict = dict(zip(self.tensor_dict.keys(), vals))
        e = ForwardEvaluation(data_set.data_type, int(global_step), idxs, yp.tolist(), yp2.tolist(), float(loss), id2answer_dict, tensor_dict=tensor_dict)
        # TODO : wy support
        return e

    @staticmethod
    def compare(yi, ypi, yp2i):
        for start, stop in yi:
            aypi = argmax(ypi)
            mask = np.zeros(yp2i.shape)
            mask[aypi[0], aypi[1]:] = np.ones([yp2i.shape[1] - aypi[1]])
            if tuple(start) == aypi and (stop[0], stop[1]-1) == argmax(yp2i * mask):
                return True
        return False

    @staticmethod
    def compare2(yi, span):
        for start, stop in yi:
            if tuple(start) == span[0] and tuple(stop) == span[1]:
                return True
        return False

    @staticmethod
    def span_f1(yi, span):
        max_f1 = 0
        for start, stop in yi:
            if start[0] == span[0][0]:
                true_span = start[1], stop[1]
                pred_span = span[0][1], span[1][1]
                f1 = span_f1(true_span, pred_span)
                max_f1 = max(f1, max_f1)
        return max_f1


class BleuEvaluation(AccuracyEvaluation):
    def __init__(self, data_type, global_step, idxs, translations,loss, bleu_score, data,summaries_data,tensor_dict=None):
        self.data_type = data_type
        self.global_step = global_step
        self.idxs = idxs
        self.num_examples = len(translations)
        self.bleu_score = bleu_score
        self.data = data
        self.loss = loss
        self.generated_answers=translations
        self.tensor_dict = None
        self.summaries_data=summaries_data
        average_bleu1 = 0.0
        average_bleu4 = 0.0
        for bleu_pair in bleu_score:
            average_bleu1+=bleu_pair[0]
            average_bleu4+=bleu_pair[1]
        average_bleu1/=len(bleu_score)
        average_bleu4/=len(bleu_score)
        combined_sol=[]
        bleu_iter=iter(self.bleu_score)
        for index,zipped_dat in enumerate(zip(data['q'],data['answerss'],translations,data['spans'],data['*x'])):
            span_based_ans = get_answer_from_span([zipped_dat[3][0][1]],[zipped_dat[3][1][1]],summaries_data,[zipped_dat[4]])
            combined_sol.append([zipped_dat[0],zipped_dat[1][1],span_based_ans,zipped_dat[2]])
            if (index+1) % 5 == 0: #index % batch size
                combined_sol.append(next(bleu_iter))
        self.dict = {'data_type': data_type,
                     'global_step': global_step,
                     #'Questions': data['q'],
                     #'Actual_answers': data['answerss'],
                     #'Generated_answers': translations,
                     'num_examples': self.num_examples,
                     'loss': self.loss,
                     #'bleu scores': self.bleu_score,
                     'OUTPUT': combined_sol,
                     'Average_BLEU_1': average_bleu1,
                     'Average_BLEU_4': average_bleu4
                      }
        if tensor_dict is not None:
            self.tensor_dict = {key: val.tolist() for key, val in tensor_dict.items()}
            for key, val in self.tensor_dict.items():
                self.dict[key] = val

        #TODO: fix this summary
        #f1_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/bleu'.format(data_type), simple_value=np.mean(self.bleu_score))])
        loss_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/loss'.format(data_type), simple_value=self.loss)])
        #acc_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/acc'.format(data_type), simple_value=self.acc)])
        self.summaries = [loss_summary]
        #self.summaries.append(f1_summary)

    def __repr__(self):
        return "{} step {}: accuracy={}".format(self.data_type, self.global_step, self.bleu_score)

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_idxs = self.idxs + other.idxs
        new_generated_answer = self.generated_answers + other.generated_answers
        new_bleus = self.bleu_score + other.bleu_score
        new_loss = (self.loss * self.num_examples + other.loss * other.num_examples) /(self.num_examples + other.num_examples)  #TODO: change this
        #ds = [self.data ,other.data]
        #new_data= {}
        for key,v in self.data.items():
            #new_data[key]= tuple(d[key] for d in ds)
            self.data[key].extend(other.data[key])
        e = BleuEvaluation(self.data_type, self.global_step, new_idxs, new_generated_answer, new_loss, new_bleus, self.data,summaries_data=self.summaries_data)
        return e



class BleuEvaluator(Evaluator):
    def __init__(self, config, model, tensor_dict=None):
        super(BleuEvaluator, self).__init__(config, model, tensor_dict=tensor_dict)
        self.loss = model.loss_task1
        if config.na:
            self.na = model.na_prob
     #softmax on decoder logits train to get the probability distribution over the vocab.        
    def get_evaluation(self, sess, batch):
        idxs, data_set = batch[0] #TODO to check why an extra tuple is being formed here

        feed_dict = self.model.get_feed_dict(data_set,False)
        #decoder_logits_train is BxWxV , take max over V
        self.model.decoder_logits_train=tf.Print(self.model.decoder_logits_train,[tf.shape(self.model.decoder_logits_train)],
                                                 message="eval decoder logits shape",summarize=20,first_n=7)
        #pred_dist = tf.nn.softmax(self.model.decoder_logits_train)
        #pred_dist= tf.Print(pred_dist,[tf.shape(pred_dist),data_set.data_type,pred_dist],message="eval pred dist",summarize=20,first_n=7)
        #take max over the vocab to get the predicted words
        #translation_corpus = tf.argmax(pred_dist, dimension = 2)
        translation_corpus = tf.argmax(self.model.decoder_logits_train, -1)
        #reference corpus  BxW
        global_step, decoder_logits_train, decoder_targets, loss,answer_lengths = sess.run(
            [self.global_step, translation_corpus, self.model.decoder_targets, self.loss,self.model.target_sequence_length],feed_dict=feed_dict)
        reference_corpus = decoder_targets
        ref_new=[]
        if self.config.mode == 'test':
            loss =0
        print ("before idx2word ref corpus",reference_corpus)
        for i in range(len(reference_corpus)):
            ref_new.append(reference_corpus[i][0:answer_lengths[i]])
        reference_corpus=ref_new
        reference_corpus=id2word_translate(reference_corpus,data_set.shared['idx2word'],data_set.shared['x'],data_set.data['*x'],self.config.pointer_network)
        print ("before idx2word trans corpus",decoder_logits_train)
        decoder_logits_train=id2word_translate(decoder_logits_train,data_set.shared['idx2word'],data_set.shared['x'],data_set.data['*x'],self.config.pointer_network)
        bleu_score = compute_bleu_new(reference_corpus,decoder_logits_train)
        #print ("bleu score 4",bleu_score[1]," bleu score 1",bleu_score[0])
        e = BleuEvaluation(data_set.data_type, int(global_step), idxs, decoder_logits_train, loss,[bleu_score],
                              data_set.data, data_set.shared['x'],tensor_dict=self.tensor_dict)
        return e

class BleuEvaluatorSpan(Evaluator):
    def __init__(self, config, model, tensor_dict=None):
        super(BleuEvaluatorSpan, self).__init__(config, model, tensor_dict=tensor_dict)
        self.loss = model.loss_task2
        if config.na:
            self.na = model.na_prob
     #softmax on decoder logits train to get the probability distribution over the vocab.
    def get_evaluation(self, sess, batch):
        idxs, data_set = batch[0] #TODO to check why an extra tuple is being formed here

        feed_dict = self.model.get_feed_dict(data_set,False)

        translated_index_start = tf.argmax(self.model.flat_yp,-1)
        translated_index_stop = tf.argmax(self.model.flat_yp2,-1)
        global_step, index_start, index_end, loss,decoder_targets = sess.run(
            [self.global_step, translated_index_start, translated_index_stop, self.loss,self.model.decoder_targets],feed_dict=feed_dict)

        translated_corpus = get_answer_from_span(index_start,index_end,data_set.shared['x'],data_set.data['*x'])
        reference_corpus = decoder_targets
        if self.config.mode == 'test':
            loss =0
        print ("before idx2word ref corpus",reference_corpus)
        reference_corpus=id2word_translate(reference_corpus,data_set.shared['idx2word'],data_set.shared['x'],data_set.data['*x'],self.config.pointer_network)
        bleu_score = compute_bleu_new(reference_corpus,translated_corpus)
        print ("bleu score 4",bleu_score[1]," bleu score 1",bleu_score[0])
        e = BleuEvaluation(data_set.data_type, int(global_step), idxs, translated_corpus, loss,[bleu_score],
                              data_set.data, data_set.shared['x'],tensor_dict=self.tensor_dict)
        return e

def get_answer_from_span(index_start,index_end,summaries,summary_id):
    #print ("index start",index_start)
    #print ("index_end",index_end)
    trans_answers=[]
    #print ("summary id",summary_id)
    #print ("summary",summaries[summary_id[0][0]][0][0])
    for i in range(len(index_start)):
        summary = summaries[summary_id[i][0]][0][0]
        trans_answers.append(summary[index_start[i]:index_end[i]+1])
    #print ("translated corpus",trans_answers)
    return trans_answers


def id2word_translate(corpus,trans_dict,summaries,summary_id,pointer_network):
    trans_corp=[]
    for answer,id in zip(corpus,summary_id):
        if pointer_network:
            translated_ans =[summaries[id[0]][0][0][position] for position in answer]
        else:
            translated_ans=[trans_dict[id] for id in answer if id!=0]
        trans_corp.append(translated_ans)
    return trans_corp

def compute_bleu_new(reference_corpus,translation_corpus):
    modified_ref_corp = []
    print ("refernce corpus",reference_corpus)
    print ("translated_corpus",translation_corpus)
    for answer in reference_corpus:
        answer = [word for word in answer if word not in ['-NULL-','</s>','.']]
        modified_ref_corp.append([answer])
    #returns BLEU1,BLEU4 scores
    #print ("modified refernce corpus new",modified_ref_corp)
    return (corpus_bleu(modified_ref_corp,translation_corpus,weights=(1,0,0,0)),corpus_bleu(modified_ref_corp,translation_corpus))

def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.
  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.
  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  print ("reference corpus",reference_corpus)
  print ("translation corpus",translation_corpus)
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  #print ("refernce corpus",reference_corpus)
  #print ("translation corpus",translation_corpus)
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  if reference_length == 0:
      reference_length=1
  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)
