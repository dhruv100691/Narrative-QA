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

# Embedding layers
input_embedding = tf.Variable(tf.random_uniform((len(char2numX), embed_size), -1.0, 1.0), name='enc_embedding')
output_embedding = tf.Variable(tf.random_uniform((len(char2numY), embed_size), -1.0, 1.0), name='dec_embedding')
date_input_embed = tf.nn.embedding_lookup(input_embedding, inputs)
date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs)

with tf.variable_scope("encoding") as encoding_scope:
    lstm_enc = tf.contrib.rnn.BasicLSTMCell(nodes)
    #encoder_output, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=date_input_embed, dtype=tf.float32)
    ((encoder_fw_outputs, encoder_bw_outputs),
     (encoder_fw_final_state, encoder_bw_final_state)) = (
        _bidirectional_dynamic_rnn(cell_fw=lstm_enc,
                                   cell_bw=lstm_enc,
                                   inputs=date_input_embed,
                                   sequence_length=tf.fill([batch_size], x_seq_length),
                                   dtype=tf.float32, time_major=False)
    )

encoder_output = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)
"""
with tf.variable_scope("decoding") as decoding_scope:
    lstm_dec = tf.contrib.rnn.BasicLSTMCell(nodes)
    dec_outputs, _ =tf.nn.dynamic_rnn(lstm_dec, inputs=date_output_embed, dtype=tf.float32,initial_state=last_state)

dec_outputs = tf.Print(dec_outputs,[tf.shape(dec_outputs)],message="dec outputs shape",first_n=5)
logits = tf.contrib.layers.fully_connected(dec_outputs, num_outputs=len(char2numY), activation_fn=None)
logits = tf.Print(logits,[tf.shape(logits)],message="Logits shape",first_n=5)
"""


def decode_with_attention(helper, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=nodes, memory=encoder_output)
        cell = tf.contrib.rnn.GRUCell(num_units=nodes)
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=nodes / 2)
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, len(char2numY), reuse=reuse)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=out_cell, helper=helper,initial_state=out_cell.zero_state(
                                                                        dtype=tf.float32, batch_size=batch_size))
        # initial_state=encoder_final_state)
        outputs = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False,impute_finished=True,maximum_iterations=y_seq_length)
        return outputs[0]

def decode(helper, scope, reuse=None, maximum_iterations=None):
    with tf.variable_scope(scope, reuse=reuse):
        decoder_cell = tf.contrib.rnn.BasicLSTMCell(nodes, state_is_tuple=True,reuse=reuse)  # hparam
        projection_layer = layers_core.Dense(len(char2numY), use_bias=False)  # hparam
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_final_state,output_layer=projection_layer)  # decoder
        final_outputs, _ ,_= tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=False, impute_finished=True,maximum_iterations=maximum_iterations)  # dynamic decoding
        return final_outputs

training_helper = tf.contrib.seq2seq.TrainingHelper(date_output_embed, tf.fill([batch_size],y_seq_length),time_major=False)
#final_outputs = decode(helper=training_helper, scope="decoder", reuse=None,)
final_outputs = decode_with_attention(helper=training_helper, scope="decoder", reuse=None)

prediction_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(output_embedding, tf.fill([batch_size], char2numY['<GO>']),char2numY['</s>'])
#final_outputs_1 = decode(helper=prediction_helper, scope="decoder", reuse=True,maximum_iterations=y_seq_length)
final_outputs_1 = decode_with_attention(helper=prediction_helper, scope="decoder", reuse=True)

with tf.name_scope("optimization"):
    # Loss function
    #loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,logits=final_outputs.rnn_output)
    loss = (tf.reduce_sum(crossent * tf.ones([batch_size, y_seq_length])) / tf.cast(batch_size, dtype=tf.float32))
    tf.summary.scalar("loss",loss)
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('/Users/dhruv100691/Documents/cs546/CS-546--Narrative-QA/bi-att-flow-dev', graph=tf.get_default_graph())

print("input shape",inputs.get_shape().as_list())
print("input embedding shape",date_input_embed.get_shape().as_list())
print("output embedding shape",date_output_embed.get_shape().as_list())
print("encoder output shape",encoder_output.get_shape().as_list())
print("final decoder output shape",final_outputs.rnn_output.get_shape().as_list())

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
sess.run(tf.global_variables_initializer())
epochs = 10
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
        _, batch_loss, batch_logits,summary_eval = sess.run([optimizer, loss, final_outputs.rnn_output,merged_summary],
            feed_dict = {inputs: source_batch,
             outputs: target_batch[:, :-1],
             targets: target_batch[:, 1:-1]})
        writer.add_summary(summary_eval)
    accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:,1:-1])
    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss,
                                                                                          accuracy, time.time() - start_time))

source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))
batch_logits = sess.run(final_outputs_1.rnn_output,feed_dict={inputs: source_batch,outputs:target_batch})
prediction = batch_logits.argmax(axis=-1)
print('Accuracy on test set is: {:>6.3f}'.format(np.mean(prediction == target_batch[:,1:-1])))

num_preds = 10
source_chars = [[num2charX[l] for l in sent if num2charX[l]!="<PAD>"] for sent in source_batch[:num_preds]]
dest_chars = [[num2charY[l] for l in sent] for sent in prediction[:num_preds, :]]

for date_in, date_out in zip(source_chars, dest_chars):
    print(''.join(date_in)+' => '+''.join(date_out))