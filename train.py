from preprocess import read_data, full_process
from models import motor
import argparse
import yaml
import pickle
import tensorflow as tf


def train(config_path):

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    en_url = config['en_url']
    fr_url = config['fr_url']
    en_input = read_data(en_url)
    fr_output = read_data(fr_url)

    process_input, process_output, token_en, token_fr = full_process(en_input, fr_output, length=None, type_pad="post")

    with open('outputs/tokenizer.pkl', 'wb') as f:
        pickle.dump(token_fr, f)

    dict_en_size = len(token_en.word_index)
    dict_fr_size = len(token_fr.word_index)

    
    type_model = config['type_model']
    if type_model != "Rnn":
      process_input = process_input.reshape(process_input.shape[0], process_input.shape[1])
    input_shape = process_input.shape

    learning_rate = config['learning_rate']
    model = motor(input_shape, dict_en_size, dict_fr_size, type_model, learning_rate)

    number_sample = config['number_sample']
    batch_size = config['batch_size']
    epochs = config['epochs']
    validation_split = config['validation_split']
    type_device = config['type_device']
    
    if type_device == "GPU":
      with tf.device('/GPU:0'):
        history = model.fit(
          process_input[:number_sample],
          process_output[:number_sample],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=validation_split,
          shuffle=True
          )
    else:
      history = model.fit(
          process_input[:number_sample],
          process_output[:number_sample],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=validation_split,
          shuffle=True
          )
    
    model.save('outputs/model.h5')


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Process some URLs.')
   parser.add_argument('--config_path', type=str, help='The URL for configuration train')
   args = parser.parse_args()

   config_path = args.config_path
   train(config_path)