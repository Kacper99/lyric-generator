import tensorflow as tf

import numpy as np
import os
import argparse
import time

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


parser = argparse.ArgumentParser('Generate lyrics')
parser.add_argument('-s', '--skip', help='Skip training', action='store_true')

args = parser.parse_args()

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





text = open('data/arctic_monkeys-corpus.txt', 'rb').read().decode(encoding='utf-8')

# Get all the unique characters in the text
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

# Processing the Text
# Mapping an ID to each character
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

print('{')
for char in char2idx:
    print('    {:4s}: {:3d},'.format(char, char2idx[char]))
print('\n}')

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Create training sets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

dataset = sequences.map(split_input_target)
# for input_example, output_example in dataset.take(1):
#     print('Input data: ', ''.join(idx2char[input_example]))
#     print('Output data: ', ''.join(idx2char[output_example]))

BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
# print(dataset)


# Building the model
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

# Configure checkpointspython
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

if not args.skip:
    model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    model.summary()

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(optimizer='adam', loss=loss)

    # Execute training
    EPOCHS = 50
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


# Generate the text
# tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

# model.save_weights('weights/am_weights1')
# model.save('models/am_model1.h5')

def generate_text(model, start_string):
    # Number of characters to generate
    num_generate = 1193

    input_ids = [char2idx[s] for s in start_string]
    input_ids = tf.expand_dims(input_ids, 0)

    text_generated = []

    # Low - results in more predictable text
    # High - more surprising text
    temperature = 0.9

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_ids)
        predictions = tf.squeeze(predictions, 0)

        # Use a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # Pass the predicted character as the next input to the model
        input_ids = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)


print(generate_text(model, '[Verse 1] '))

