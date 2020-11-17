#Implementation of Bahdanau's Self Attention
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

class Attention(Layer):
    def __init__(self, units):
        super( Attention, self).__init__()
        self.W1= Dense(units)  # encoder output
        self.W2= Dense(units)  # Decoder hidden
        self.V= Dense(1)
    
    def call(self, query, values):
        #calculate the Attention score
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score= self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights= tf.nn.softmax(score, axis=1)
        
         #context_vector 
        context_vector= attention_weights * values
       
        #Computes the sum of elements across dimensions of a tensor
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights