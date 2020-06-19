import tensorflow as tf

import numpy as np
import os
import time

def split_input_target(chunk):
    """
    For each sequence:
    Duplicate and shift it
    This will from the input and target text

    :param chunk: The chunk or sequence to shift
    :return: The input text and the target text
    """
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


text = open('data/blossoms-corpus.txt').read()

# Get all the unique characters in the text
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

# Processing the Text
# Mapping an ID to each character
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

# Create training sets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

dataset = sequences.map(split_input_target)
# for input_example, output_example in dataset.take(1):
#     print('Input data: ', ''.join(idx2char[input_example]))
#     print('Output data: ', ''.join(idx2char[output_example]))

BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
# print(dataset)

# Building the model
