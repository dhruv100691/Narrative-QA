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
num2charY = dict(zip(char2numY.values(), char2numY.keys()))

y = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y]
print(''.join([num2charY[y_] for y_ in y[4]]))
y = np.array(y)

x_seq_length = len(x[0])
y_seq_length = len(y[0])- 1

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
    _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=date_input_embed, dtype=tf.float32)

with tf.variable_scope("decoding") as decoding_scope:
    lstm_dec = tf.contrib.rnn.BasicLSTMCell(nodes)
    dec_outputs, _ =tf.nn.dynamic_rnn(lstm_dec, inputs=date_output_embed, dtype=tf.float32,initial_state=last_state)

dec_outputs = tf.Print(dec_outputs,[tf.shape(dec_outputs)],message="dec outputs shape",first_n=5)
logits = tf.contrib.layers.fully_connected(dec_outputs, num_outputs=len(char2numY), activation_fn=None)
logits = tf.Print(logits,[tf.shape(logits)],message="Logits shape",first_n=5)

with tf.name_scope("optimization"):
    # Loss function
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

print(inputs.get_shape().as_list())
print(date_input_embed.get_shape().as_list())
print(last_state[0].get_shape().as_list())
print(dec_outputs.get_shape().as_list())
print(logits.get_shape().as_list())

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
sess.run(tf.global_variables_initializer())
epochs = 1
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
        _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
            feed_dict = {inputs: source_batch,
             outputs: target_batch[:, :-1],
             targets: target_batch[:, 1:]})
    accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:,1:])
    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss,
                                                                                          accuracy, time.time() - start_time))

source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))

dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']
#print ("dec input",dec_input)
for i in range(y_seq_length):
    batch_logits = sess.run(logits,
                            feed_dict={inputs: source_batch,
                                       outputs: dec_input})
    prediction = batch_logits[:, -1].argmax(axis=-1)
    dec_input = np.hstack([dec_input, prediction[:, None]])
#print ("stacked decoder output",dec_input)

print('Accuracy on test set is: {:>6.3f}'.format(np.mean(dec_input == target_batch)))

num_preds = 10
source_chars = [[num2charX[l] for l in sent if num2charX[l]!="<PAD>"] for sent in source_batch[:num_preds]]
dest_chars = [[num2charY[l] for l in sent] for sent in dec_input[:num_preds, 1:]]

for date_in, date_out in zip(source_chars, dest_chars):
    print(''.join(date_in)+' => '+''.join(date_out))
