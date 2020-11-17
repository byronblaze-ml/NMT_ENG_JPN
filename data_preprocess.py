# Preprocess data for further processing

def read_text(text):
  
  with open(text, 'r', encoding="utf8") as txt:
    data = txt.read()
  
  return data.split('\n')

def separate_langs(raw_data):

  src_sent = []
  tgt_sent = []

  for word in raw_data:
    src_sent.append(word.split('\t')[:-1][0])
    tgt_sent.append(word.split('\t')[:-1][1])
  
  return src_sent, tgt_sent

def preprocess(sentence):

  num_digits= str.maketrans('','', digits)
  punc = str.maketrans('', '', string.punctuation)  

  sentence = sentence.lower()
  sentence = re.sub(" +", " ", sentence)
  sentence = re.sub("'", "". sentence)
  sentence = sentence.translate(num_digits)
  sentence = sentence.translate(punc)
  sentence = re.sub(r"([?.!])", r" \1 ", sentence)
  sentence = sentence.rstrip()/strip()
  sentence = 'start_ '+ sentence + ' _end'

  return sentence


def main(text):
  
  raw_data = read_text(text)
  src_sent, tgt_sent = separate_langs(raw_data)

  src_sent_clean = [preprocess(sent) for sent in src_sent]
  tgt_sent_clean = [preprocess(sent) for sent in tgt_sent]

  return src_sent_clean, tgt_sent_clean
