from keras.models import Model, Sequential
from keras.layers import SimpleRNN, Dense, TimeDistributed, Dropout, LSTM, Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


def motor(input_shape, 
          dict_en_size, 
          dict_fr_size, 
          type_model,
          n_neurons_embedding,
          n_neurons_rnn,
          n_neurons_lstm,
          n_neurons_timedistributed,
          learning_rate):
    
    model = Sequential()
    
    if type_model == "Rnn":
        
            model.add(SimpleRNN(n_neurons_rnn, input_shape=input_shape[1:], return_sequences=True))
            model.add(TimeDistributed(Dense(n_neurons_timedistributed, activation='relu')))
            model.add(Dropout(0.5))
            model.add(TimeDistributed(Dense(dict_fr_size, activation='softmax'))) 
    
    elif type_model == "Rnn_Embd":

            model.add(Embedding(dict_en_size, n_neurons_embedding, input_length=input_shape[1], input_shape=input_shape[1:]))
            model.add(SimpleRNN(n_neurons_rnn, return_sequences=True))    
            model.add(TimeDistributed(Dense(n_neurons_timedistributed, activation='relu')))
            model.add(Dropout(0.5))
            model.add(TimeDistributed(Dense(dict_fr_size, activation='softmax')))
            
    elif type_model == "Lstm":
        
        model.add(LSTM(n_neurons_lstm, input_shape=input_shape[1:], return_sequences=True))
        model.add(TimeDistributed(Dense(n_neurons_timedistributed, activation='relu')))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(dict_fr_size, activation='softmax'))) 

    elif type_model == "Lstm_Embd":

        model.add(Embedding(dict_en_size, n_neurons_embedding, input_length=input_shape[1], input_shape=input_shape[1:]))
        model.add(LSTM(n_neurons_lstm, return_sequences=True))    
        model.add(TimeDistributed(Dense(n_neurons_timedistributed, activation='relu')))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(dict_fr_size, activation='softmax'))) 

    print(model.summary())
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    
    return model