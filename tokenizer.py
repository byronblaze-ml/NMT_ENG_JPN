#create source and target sentence tokenizer

from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences


def source_tokenize(text):

  src_sent_tokenizer= Tokenizer(filters='')
  src_sent_tokenizer.fit_on_texts(text)
  src_tensor = src_sent_tokenizer.texts_to_sequences(text)
  src_tensor= pad_sequences(src_tensor,padding='post' )

  return src_tensor, src_sent_tokenizer

def target_tokenize(text):

  tgt_sent_tokenizer= Tokenizer(filters='')
  tgt_sent_tokenizer.fit_on_texts(text)
  tgt_tensor = tgt_sent_tokenizer.texts_to_sequences(text)
  tgt_tensor= pad_sequences(tgt_tensor, padding='post' )

  return tgt_tensor, tgt_sent_tokenizer

