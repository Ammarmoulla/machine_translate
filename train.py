from preprocess import read_data, full_process
from models import motor
import os
import argparse
from pathlib import Path
import yaml
import pickle
import tensorflow as tf
import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
import requests

BASE_DIR = Path(__file__).resolve().parent

neptune_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NWE2M2I5My1iZjFmLTRhOWItOGEyNy01YjBlYzMwZmQzNWIifQ=="
telegram_token = "6645018983:AAG2nTpOuCxwdgfMZTlxkmlBxPchFrm8fec"
chat_id = "903737895"

def send_telegram(text):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    response = requests.post(url, data=data)

def train(config_path):

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    en_url = config['en_url']
    fr_url = config['fr_url']
    en_input = read_data(en_url)
    fr_output = read_data(fr_url)

    process_input, process_output, token_en, token_fr = full_process(en_input, fr_output, length=None, type_pad="post")

    with open('outputs/tokenizer_en.pkl', 'wb') as f:
        pickle.dump(token_en, f)
    
    with open('outputs/tokenizer_fr.pkl', 'wb') as f:
        pickle.dump(token_fr, f)

    dict_en_size = len(token_en.word_index) + 1
    dict_fr_size = len(token_fr.word_index) + 1

    
    type_model = config['type_model']
    if type_model not in ["Rnn","Lstm"]:
      process_input = process_input.reshape(process_input.shape[0], process_input.shape[1])
    input_shape = process_input.shape
   
    length_vector_word = config['length_vector_word']
    n_neurons_rnn = config['n_neurons_rnn']
    n_neurons_lstm = config['n_neurons_lstm']
    n_neurons_timedistributed = config['n_neurons_timedistributed']
    learning_rate = config['learning_rate']
    model = motor(input_shape,
                  dict_en_size, 
                  dict_fr_size, 
                  type_model,
                  length_vector_word,
                  n_neurons_rnn,
                  n_neurons_lstm,
                  n_neurons_timedistributed,
                  learning_rate)

    number_sample = config['number_sample']
    batch_size = config['batch_size']
    epochs = config['epochs']
    validation_split = config['validation_split']
    type_device = config['type_device']
    

    #Monitor
    run = neptune.init_run(
    name=type_model,  
    project="ammar.mlops/translate-en-to-fr",
    api_token=neptune_token)
    url_project = run.get_url()
    
    send_telegram("The URL ML Track for model: "
                  + f"<b>{type_model}</b> 🤓"
                  + "\nPlease Use <b> VPN </b>😅 \n"
                  + f"{url_project}\n.")

    neptune_callback = NeptuneCallback(run=run,
                                       log_model_diagram=True)
    if type_device == "GPU":
      with tf.device('/GPU:0'):
        history = model.fit(
          process_input[:number_sample],
          process_output[:number_sample],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=validation_split,
          shuffle=True,
          callbacks=[neptune_callback],
          )
    else:
      history = model.fit(
          process_input[:number_sample],
          process_output[:number_sample],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=validation_split,
          shuffle=True,
          callbacks=[neptune_callback],
          )
    
    result_path = os.path.join(BASE_DIR, os.path.join("outputs", f"model_{type_model}.h5"))
    model.save(result_path)


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Process some URLs.')
   parser.add_argument('--config_path', type=str, help='The URL for configuration train')
   args = parser.parse_args()

   config_path = args.config_path
   train(config_path)

import yaml
config_path = "config.yaml"
with open(config_path, 'r') as file:
    config_para = yaml.safe_load(file)

image = config_para['image']
classesf = config_para['classesf']
weights = config_para['weights']
config = config_para['config']
scale = config_para['scale']
classes = config_para['classes']

