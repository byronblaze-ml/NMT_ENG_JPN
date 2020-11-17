import tensorflow as tf
from tensorflow.keras.layers import Dense,Embedding, GRU
from tensorflow.keras.models import Model
import attention


class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, encoder_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size= batch_size
        self.encoder_units=encoder_units
        self.embedding=Embedding(vocab_size, embedding_dim)
        self.gru= GRU(encoder_units, 
                      return_sequences=True,
                      return_state=True,
                      recurrent_initializer='glorot_uniform'
                      )
    
    def call(self, x, hidden):
        #pass the input x to the embedding layer
        x= self.embedding(x)
        # pass the embedding and the hidden state to GRU
        output, state = self.gru(x, initial_state=hidden)
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoder_units))

class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, decoder_units, batch_sz):
        super (Decoder,self).__init__()
        self.batch_sz= batch_sz
        self.decoder_units = decoder_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru= GRU(decoder_units, 
                      return_sequences= True,
                      return_state=True,
                      recurrent_initializer='glorot_uniform')
        # Fully connected layer
        self.fc= Dense(vocab_size)
        
        # attention
        self.attention = Attention(self.decoder_units)
    
    def call(self, x, hidden, encoder_output):
        
        context_vector, attention_weights = self.attention(hidden,      
                                                    encoder_output)
        
        # pass output sequnece thru the input layers
        x= self.embedding(x)
        
        # concatenate context vector and embedding for output sequence
        x= tf.concat([tf.expand_dims( context_vector, 1), x], 
                                      axis=-1)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        # output shape == (batch_size * 1, hidden_size)
        output= tf.reshape(output, (-1, output.shape[2]))
        
        # pass the output thru Fc layers
        x= self.fc(output)
        return x, state, attention_weights