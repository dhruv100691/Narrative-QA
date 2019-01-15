import numpy as np
#import matplotlib.pyplot as plt

import random
import json
import os
import time

from faker import Faker
import babel
from babel.dates import format_date

import tensorflow as tf

import tensorflow.contrib.legacy_seq2seq as seq2seq
#from utilities import show_graph

from sklearn.model_selection import train_test_split
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as _bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn import _transpose_batch_time
from tensorflow.python.client import timeline

fake = Faker()
fake.seed(42)
random.seed(42)

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY',
           ]

# change this if you want it to work with only a single language
LOCALES = babel.localedata.locale_identifiers()
LOCALES = [lang for lang in LOCALES if 'en' in str(lang)]

def create_date():
    """
        Creates some fake dates
        :returns: tuple containing
                  1. human formatted string
                  2. machine formatted string
                  3. date object.
    """
    dt = fake.date_object()

    # wrapping this in a try catch because
    # the locale 'vo' and format 'full' will fail
    try:
        human = format_date(dt,
                            format=random.choice(FORMATS),
                            locale=random.choice(LOCALES))

        case_change = random.randint(0,3) # 1/2 chance of case change
        if case_change == 1:
            human = human.upper()
        elif case_change == 2:
            human = human.lower()

        machine = dt.isoformat()
    except AttributeError as e:
        return None, None, None

    return human, machine #, dt

data = [create_date() for _ in range(50000)]

x = [x for x, y in data]
y = [y for x, y in data]

u_characters = set(' '.join(x))
char2numX = dict(zip(u_characters, range(len(u_characters))))

u_characters = set(' '.join(y))
char2numY = dict(zip(u_characters, range(len(u_characters))))

char2numX['<PAD>'] = len(char2numX)
num2charX = dict(zip(char2numX.values(), char2numX.keys()))
max_len = max([len(date) for date in x])

x = [[char2numX['<PAD>']]*(max_len - len(date)) +[char2numX[x_] for x_ in date] for date in x]
print (x[4])
print(''.join([num2charX[x_] for x_ in x[4]]))
x = np.array(x)

char2numY['<GO>'] = len(char2numY)
char2numY['</s>'] = len(char2numY)
num2charY = dict(zip(char2numY.values(), char2numY.keys()))

y = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y]
for each in y:
    each.append(char2numY['</s>'])
print(''.join([num2charY[y_] for y_ in y[0]]))
y = np.array(y)

x_seq_length = len(x[0])
y_seq_length = len(y[0])- 2

print ("lengths",x_seq_length,y_seq_length,len(char2numY))

def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
#     from IPython.core.debugger import Tracer; Tracer()()
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start+batch_size], y[start:start+batch_size]
        start += batch_size

batch_size = 128
nodes = 32
embed_size = 10

#works with tensorflow 1.11.0
tf.reset_default_graph()
sess = tf.Session()

# Tensor where we will feed the data into graph
inputs = tf.placeholder(tf.int32, (None, x_seq_length), 'inputs')
outputs = tf.placeholder(tf.int32, (None, None), 'output')
targets = tf.placeholder(tf.int32, (None, None), 'targets')
is_training = tf.placeholder(tf.bool)

# Embedding layers
input_embedding = tf.Variable(tf.random_uniform((len(char2numX), embed_size), -1.0, 1.0), name='enc_embedding')
output_embedding = tf.Variable(tf.random_uniform((len(char2numY), embed_size), -1.0, 1.0), name='dec_embedding')
date_input_embed = tf.nn.embedding_lookup(input_embedding, inputs)
date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs)

with tf.variable_scope("encoding") as encoding_scope:
    lstm_enc = tf.contrib.rnn.BasicLSTMCell(nodes,state_is_tuple=True)
    #encoder_output, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=date_input_embed, dtype=tf.float32)
    ((encoder_fw_outputs,encoder_bw_outputs),
     (encoder_fw_final_state,encoder_bw_final_state)) = (
        _bidirectional_dynamic_rnn(cell_fw=lstm_enc,
                                        cell_bw=lstm_enc,
                                        inputs=date_input_embed,
                                        sequence_length=tf.fill([batch_size],x_seq_length),
                                        dtype=tf.float32, time_major=False)
                                    )

encoder_output = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_final_state_c,h=encoder_final_state_h)

#need to put this in the same variable scope and use get variable
W = tf.Variable(tf.random_uniform([2*nodes, len(char2numY)], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([len(char2numY)]), dtype=tf.float32)

W_att_dec = tf.Variable(tf.random_uniform([2*nodes, 2*nodes], -1, 1), dtype=tf.float32)
W_att_enc = tf.Variable(tf.random_uniform([2*nodes, 2*nodes], -1, 1), dtype=tf.float32)
W_att_enc1 = tf.Variable(tf.random_uniform([1,1,2*nodes, 2*nodes], -1, 1), dtype=tf.float32)

b_att = tf.Variable(tf.zeros([batch_size, 2*nodes]), dtype=tf.float32)
v_blend = tf.Variable(tf.random_uniform([2*nodes, 2*nodes], -1, 1), dtype=tf.float32)

eos_time_slice = tf.fill([batch_size], char2numY['</s>'] , name='EOS')
pad_time_slice = tf.fill([batch_size], char2numX['<PAD>'], name='PAD')
eos_step_embedded = tf.nn.embedding_lookup(output_embedding, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(input_embedding, pad_time_slice)

decoder_lengths = tf.fill([batch_size],y_seq_length)
decoder_cell = tf.contrib.rnn.BasicLSTMCell(2*nodes,state_is_tuple=True) #doesnt work without the factor of 2??

'''Loop transition function is a mapping (time, previous_cell_output, previous_cell_state, previous_loop_state) -> 
(elements_finished, input, cell_state, output, loop_state).
 It is called before RNNCell to prepare its inputs and state. Everything is a Tensor except for initial call at time=0 
 when everything is None (except time).'''

def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    initial_input = tf.concat([eos_step_embedded,encoder_final_state_h],1)
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,initial_input,initial_cell_state,
            initial_cell_output,initial_loop_state)

encoder_output1 = tf.expand_dims(encoder_output, axis=2)

def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        prediction = tf.argmax(output_logits, axis=1)
        #compute Badhanau style attention
        '''
        dec_portion = tf.matmul(previous_output, W_att_dec)
        enc_portions = []
        for enc_index in range(x_seq_length):
            enc_portion = tf.matmul(encoder_output[:,enc_index],W_att_enc)
            raw_blend = tf.nn.elu(enc_portion + dec_portion + b_att)
            scaled_blend = tf.reduce_sum(tf.matmul(raw_blend, v_blend),-1)  # B x 1
            enc_portions.append(scaled_blend)

        enc_predistribution = tf.transpose(tf.stack(enc_portions)) #bacth_size*max_seq_length
        alphas_enc = tf.nn.softmax(enc_predistribution,1)
        alpha_enc_sum = tf.reduce_sum(alphas_enc,1)
        alphas_enc = alphas_enc / tf.reshape(alpha_enc_sum,(-1,1)) #bacth_size*max_seq_length

        context_vector = tf.tensordot(alphas_enc,tf.transpose(encoder_output,perm=[1,2,0]),axes=[[1], [0]])
        context_vector = tf.reshape(tf.slice(context_vector,[0,0,0],[0,-1,-1]),(batch_size,-1))

        next_input = tf.cond(is_training,lambda : tf.concat([tf.reshape(date_output_embed[:,time],(batch_size,embed_size)),context_vector],1) ,
                             lambda : tf.concat([tf.nn.embedding_lookup(output_embedding, prediction),context_vector],1))

        #next_input = tf.nn.embedding_lookup(output_embedding, prediction)
        '''
        encoder_features = tf.nn.conv2d(encoder_output1, W_att_enc1, [1, 1, 1, 1], "SAME")  # shape (batch_size,max_enc_steps,1,attention_vec_size)
        dec_portion = tf.matmul(previous_output, W_att_dec)
        decoder_features = tf.expand_dims(tf.expand_dims(dec_portion, 1), 1)  # reshape to (batch_size, 1, 1, attention_vec_size)
        e_not_masked = tf.reduce_sum(v_blend * tf.nn.tanh(encoder_features + decoder_features), [2, 3])  # calculate e, (batch_size, max_enc_steps)
        masked_e = tf.nn.softmax(e_not_masked)  # (batch_size, max_enc_steps)
        masked_sums = tf.reduce_sum(masked_e, axis=1)  # shape (batch_size)
        attn_dist = masked_e / tf.reshape(masked_sums, [-1, 1])
        context_vector = tf.reduce_sum(tf.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_output1,[1, 2])  # shape (batch_size, attn_size).
        context_vector = tf.reshape(context_vector, [-1, 2 * nodes])
        next_input = tf.cond(is_training, lambda: tf.concat([tf.reshape(date_output_embed[:, time], (batch_size, embed_size)), context_vector], 1),
                             lambda: tf.concat([tf.nn.embedding_lookup(output_embedding, prediction), context_vector],1))

        return next_input

    elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
    # defining if corresponding sequence has ended
    finished = tf.reduce_all(elements_finished)  # -> boolean scalar
    input = tf.cond(finished, lambda: tf.concat([pad_step_embedded,encoder_final_state_h],1), get_next_input)
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished,input,state,output,loop_state)

def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()

#To do output projection, we have to temporarilly flatten decoder_outputs from [max_steps, batch_size, hidden_dim] to
#  [max_steps*batch_size, hidden_dim], as tf.matmul needs rank-2 tensors at most.
decoder_max_steps,decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps,decoder_batch_size, len(char2numY)))
#decoder_logits = tf.transpose(decoder_logits,perm=[1,0,2])
decoder_logits = _transpose_batch_time(decoder_logits)
decoder_prediction = tf.argmax(decoder_logits, -1)

with tf.name_scope("optimization"):
    # Loss function
    #loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,logits=decoder_logits)
    loss = (tf.reduce_sum(crossent * tf.ones([batch_size, y_seq_length])) / tf.cast(batch_size, dtype=tf.float32))
    tf.summary.scalar("loss",loss)
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

merged_summary = tf.summary.merge_all()
#writer = tf.summary.FileWriter('/Users/dhruv100691/Documents/cs546/CS-546--Narrative-QA/bi-att-flow-dev', graph=tf.get_default_graph())

print("input shape",inputs.get_shape().as_list())
print("input embedding shape",date_input_embed.get_shape().as_list())
print("output embedding shape",date_output_embed.get_shape().as_list())
print("encoder output shape",encoder_output.get_shape().as_list())
print("last state shape",encoder_final_state[0].get_shape().as_list())
print("final decoder output shape",decoder_logits.get_shape().as_list())

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
sess.run(tf.global_variables_initializer())
epochs = 10
# add additional options to trace the session execution
#options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#run_metadata = tf.RunMetadata()
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
        _, batch_loss, batch_logits,summary_eval = sess.run([optimizer, loss, decoder_prediction,merged_summary],
            feed_dict = {inputs: source_batch,
             outputs: target_batch[:, :-1],
             targets: target_batch[:, 1:-1],
             is_training : True})
        #writer.add_summary(summary_eval)
    accuracy = np.mean(batch_logits == target_batch[:,1:-1])
    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss,
                                                                                          accuracy, time.time() - start_time))

source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))
batch_logits = sess.run(decoder_prediction,feed_dict={inputs: source_batch,outputs:target_batch,is_training:False})
print('Accuracy on test set is: {:>6.3f}'.format(np.mean(batch_logits == target_batch[:,1:-1])))

num_preds = 10
source_chars = [[num2charX[l] for l in sent if num2charX[l]!="<PAD>"] for sent in source_batch[:num_preds]]
dest_chars = [[num2charY[l] for l in sent] for sent in batch_logits[:num_preds, :]]

for date_in, date_out in zip(source_chars, dest_chars):
    print(''.join(date_in)+' => '+''.join(date_out))