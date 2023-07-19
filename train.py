from preprocess import read_data, full_process
from models import motor
import argparse
import yaml
import pickle
import tensorflow as tf
import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback

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
    n_neurons_embedding = config['n_neurons_embedding']
    n_neurons_rnn = config['n_neurons_rnn']
    n_neurons_lstm = config['n_neurons_lstm']
    n_neurons_timedistributed = config['n_neurons_timedistributed']
    learning_rate = config['learning_rate']
    model = motor(input_shape,
                  length_vector_word,
                  dict_en_size, 
                  dict_fr_size, 
                  type_model,
                  n_neurons_embedding,
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
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NWE2M2I5My1iZjFmLTRhOWItOGEyNy01YjBlYzMwZmQzNWIifQ==",
    )


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
    
    model.save(f'outputs/model_{type_model}.h5')


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Process some URLs.')
   parser.add_argument('--config_path', type=str, help='The URL for configuration train')
   args = parser.parse_args()

   config_path = args.config_path
   train(config_path)