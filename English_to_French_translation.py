import tensorflow

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64
epochs = 100
latent_dim = 256
num_samples = 20000

data_path = 'fra.txt'

input_texts = []
output_texts = []
input_characters = set()
output_characters = set()

with open(data_path, 'r', encoding = 'utf-8') as f:
    lines = f.read().split('\n')
for line in lines[:min(num_samples,len(lines)-1)]:
    input_text, output_text, _ = line.split('\t')
    
    # We use 'tab' as the 'start sequence' character
    # for the targets, and '\n' as the 'end sequence' character.
    
    output_text = '\t' + output_text + '\n'
    input_texts.append(input_text)
    output_texts.append(output_text)

    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in output_text:
        if char not in output_characters:
            output_characters.add(char)


input_characters = sorted(list(input_characters))
output_characters = sorted(list(output_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(output_characters)
max_encoder_seq_length = max([len(text) for text in input_texts])
max_decoder_seq_length = max([len(text) for text in output_texts])

print("Number of Samples:", len(input_texts))
print('Number of unique input Tokens:' , num_encoder_tokens)
print('Number of unique output Tokens:' , num_decoder_tokens)
print('Max sequence length for inputs:' , max_encoder_seq_length)
print('Max sequence length for outputs:' , max_decoder_seq_length)


input_token_index = dict(
    [(char,i) for i,char in enumerate(input_characters)])

output_token_index = dict(
    [(char,i) for i,char in enumerate(output_characters)])

encoder_input_data = np.zeros(
    (len(input_texts),max_encoder_seq_length,num_encoder_tokens),dtype = 'float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length , num_decoder_tokens), dtype = 'float32')
decoder_output_data = np.zeros(
    (len(input_texts), max_decoder_seq_length,num_decoder_tokens) , dtype = 'float32')




for i,(input_text,output_text) in enumerate(zip(input_texts,output_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i,t,input_token_index[char]] = 1.
    encoder_input_data[i,t+1:,input_token_index[' ']] = 1.
    
    for t, char in enumerate(output_text):
        # decoder_output_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i,t,output_token_index[char]] = 1.
        if t>0:
            # decoder_output_data will be ahead by one timestep
            # and will not include the start character
            decoder_output_data[i,t-1,output_token_index[char]] = 1.
    decoder_input_data[i,t+1:, output_token_index[' ']] = 1.
    decoder_output_data[i , t: , output_token_index[' ']] = 1.


# define an input sequence and process it:
encoder_inputs = Input(shape = (None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state= True)
encoder_outputs,state_h,state_c = encoder(encoder_inputs)
# we discard 'encoder_outputs' and only keep the states.
encoder_states = [state_h,state_c]

# set up the decoder , using encoder_states as initial states:
decoder_inputs= Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim,return_sequences = True, return_state = True)
decoder_outputs, _ ,_ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens,activation = 'softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# Define the model that will turn
# 'encoder_input_data' & 'decoder_input_data' into 'decoder_output_data'
model = Model([encoder_inputs,decoder_inputs], decoder_outputs)

#Run training:
model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.fit([encoder_input_data,decoder_input_data], decoder_output_data,
         batch_size=batch_size,
         epochs = epochs,
         validation_split=0.2)




# define sampling models:
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape = (latent_dim,))
decoder_state_input_c = Input(shape = (latent_dim,))
decoder_input_states = [decoder_state_input_h,decoder_state_input_c]

decoder_outputs,state_h,state_c = decoder_lstm(decoder_inputs, initial_state=decoder_input_states)
decoder_states = [state_h,state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs]+decoder_input_states,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to something readable.
reverse_input_char_index = dict(
    (i,char) for char,i in input_token_index.items())
reverse_output_char_index = dict(
    (i,char) for char,i in output_token_index.items())

def decode_sequence(input_seq):
    states_value= encoder_model.predict(input_seq)
    output_seq = np.zeros((1,1,num_decoder_tokens))
    output_seq[0,0,output_token_index['\t']] = 1
    
    stop_condition = False
    decoded_sentences = ''
    while not stop_condition:
        output_tokens,h,c = decoder_model.predict(
            [output_seq]+ states_value)
        
        sampled_token_index = np.argmax(output_tokens[0,-1, :])
        sampled_char = reverse_output_char_index[sampled_token_index]
        decoded_sentences += sampled_char

        if(sampled_char == '\n' or len(decoded_sentences) > max_decoder_seq_length):
            stop_condition = True
        
        #update the target sequence (of length 1):
        output_seq = np.zeros((1,1,num_decoder_tokens))
        output_seq[0,0,sampled_token_index] = 1
        
        states_value = [h,c]
    return decoded_sentences

for seq_index in range(20):
    # take one sequence for trying out decoding:
    input_seq = encoder_input_data[seq_index:seq_index+1]
    decoded_sentences = decode_sequence(input_seq)
    print(' English to French Translation ')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:' , decoded_sentences)




