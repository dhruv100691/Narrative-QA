import tensorflow as tf

from basic.model_nqa import Model
from my.tensorflow import average_gradients
import numpy as np


class Trainer(object):
    def __init__(self, config, model):
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.opt = tf.train.AdamOptimizer(config.init_lr)
        self.loss = model.get_loss()
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.summary = model.summary
        self.grads = self.opt.compute_gradients(self.loss, var_list=self.var_list)
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    def get_train_op(self):
        return self.train_op

    def step(self, sess, batch, get_summary=False):
        assert isinstance(sess, tf.Session)
        _, ds = batch
        feed_dict = self.model.get_feed_dict(ds, True)
        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op


class MultiGPUTrainer(object):
    def __init__(self, config, models):
        model = models[0]
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.opt = tf.train.AdamOptimizer(config.init_lr)
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.summary = model.summary
        self.models = models
        losses_task1 = []
        losses_task2 = []
        grads_list_task1 = []
        grads_list_task2 = []
        #print("VAR LIST",self.var_list) # TODO: Check is this should not be None?
        for gpu_idx, model in enumerate(models):
            with tf.name_scope("grads_{}".format(gpu_idx)), tf.device("/{}:{}".format(config.device_type, gpu_idx)):
                loss_task1 = model.get_loss_task1()
                loss_task2 = model.get_loss_task2()
                grads_task1 = self.opt.compute_gradients(loss_task1, var_list=self.var_list)
                grads_task2 = self.opt.compute_gradients(loss_task2, var_list=self.var_list)
                losses_task1.append(loss_task1)
                losses_task2.append(loss_task2)
                grads_list_task1.append(grads_task1)
                grads_list_task2.append(grads_task2)


        self.loss_task1 = tf.add_n(losses_task1)/len(losses_task1)
        self.loss_task2 = tf.add_n(losses_task2)/len(losses_task2)
        self.grads_task1 = average_gradients(grads_list_task1)
        self.grads_task2 = average_gradients(grads_list_task2)

        self.train_op_task1 = self.opt.apply_gradients(self.grads_task1, global_step=self.global_step)
        self.train_op_task2 = self.opt.apply_gradients(self.grads_task2, global_step=self.global_step)


    def step(self, sess, batches, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = {}
        for batch, model in zip(batches, self.models):
            _, ds = batch
            feed_dict.update(model.get_feed_dict(ds, True))
        if get_summary:
            if np.random.rand() < 1.0:
                loss, summary, train_op= \
                    sess.run([self.loss_task2, self.summary, self.train_op_task2], feed_dict=feed_dict)
            else:
                loss, summary, train_op = \
                    sess.run([self.loss_task2, self.summary, self.train_op_task2], feed_dict=feed_dict)
        else:
            if np.random.rand() < 1.0:
                loss, train_op = sess.run([self.loss_task2, self.train_op_task2], feed_dict=feed_dict)
            else:
                loss, train_op = sess.run([self.loss_task2, self.train_op_task2], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op
