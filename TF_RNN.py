import tensorflow as tf

import numpy as np
import os
import time


path_to_file = "C:\\Users\\Danno3321\\Desktop\\Python NN\\HardrockMetal.txt"


# Read the data found at the file path
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))

vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))

# Creating a mapping from unique characters in our data to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)



sequences = char_dataset.batch(seq_length+1, drop_remainder=True)



# A function to take a given piece of text and split its inputs from its targets, to help us learn to predict the next character
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars. Sourced from our read data
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units to use
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

# Use the above model making function to create a model based on the characteristics of our data
model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

# Print some information about our model
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()


# Prepare a test-run of the model
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# Train our model over a pre-set number of epochs
EPOCHS=10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# Revert to the last version of the model to prepare for text generation
tf.train.latest_checkpoint(checkpoint_dir)


# Construct our model based on its last checkpoint
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

# Given a trained model, a starting string, and optionally a temperature value (included for ease of testing the temperature value's impact),
# generate a set number of characters based on the starting string and the trained model's predictions.
def generate_text(model, start_string, temp = 1.0):
  # Number of characters to generate. Worst case, this method will generate twice this number should it take that long to end its last line
  num_generate = 1000

  # Convert our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Store the results of the text generation into a string
  text_generated = []

  # Use the temperature value provided. Defaults to 1.0, which allows some variance from the predictions.
  # Higher values allow wider variance, lower values force more exact generations
  temperature = temp

  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  # Check the last generated character to try to end on a new line.
  # In the worst case, we fail to generate a new line within num_generate characters
  # and end text generation anyways to avoid infinitely looping.
  last_char = text_generated[-1]
  for i in range (num_generate):
    if last_char == '\n':
      return (start_string + ''.join(text_generated))
    predictions = model(input_eval)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the character returned by the model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

    # We pass the predicted character as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(idx2char[predicted_id])
    last_char = text_generated[-1]
  return (start_string + ''.join(text_generated))


# Generate text based on the final trained model and a starting string.
# Continue to do so until anything other than y is inputted by the user
x = True
cont = ""
while x is True:
    print(generate_text(model, start_string=u"[INTRO]\n"))
    cont = ""
    cont = input("Continue? y/n")
    if (cont == "y"):
        continue
    else:
        x = False





