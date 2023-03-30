import transformers
import json
import pandas as pd
import numpy
#tokenized_questions, tokenized_answers = [], []



with open('ehealthforumQAs.json') as f1:
  ehealth=json.load(f1)
with open('icliniqQAs.json') as f2:
  icliniq=json.load(f2)
with open('questionDoctorQAs.json') as f3:
  questiondoctor=json.load(f3)
with open('webmdQAs.json') as f4:
  webmd=json.load(f4)




def extract_answer_question_tags(json_data):
  questions=[]
  answers=[]
  tags=[]
  for i in json_data:
    questions.append(i['question'])
    answers.append(i['answer'])
    tags.append(i['tags'])
  return questions,answers,tags


all_questions=[]
all_answers=[]
all_tags=[]

for i in [ehealth,icliniq,questiondoctor,webmd]:
  questions,answers,tags=extract_answer_question_tags(i)
  all_questions.extend(questions)
  all_answers.extend(answers)
  all_tags.extend(tags)
print(len(all_questions),len(all_answers),len(all_tags))

#entire dataset
all_data=pd.DataFrame({'questions':all_questions,'answers':all_answers,'tags':all_tags})
all_data.to_csv('all_data.csv',index=False)


#displaying the medical data.
print(all_data)





#preparing the negative labelled dataset
from tqdm.notebook import tqdm
#tqdm.pandas()


#negative_labels=df.apply(lambda x: pd.Series([x.question,extract_negative_samples(x.question,x.tags).answer.values[0],x.tags]),axis=1)
#negative_labels['label']=-1.0
#negative_labels.columns=['question','answer','tags','label']
#print(negative_labels.head())

#positive labels for the entire dataset
#all_dataset=df[['question','answer','tags']]
all_dataset['label']=1.0

#all_data_with_labels=pd.concat([all_dataset,negative_labels],axis=0)
print(all_dataset)



import re
#preprocessing questions and answers.
def decontractions(phrase):
    """decontracted takes text and convert contractions into natural form.
     ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)

    return phrase






def preprocess(text):
    # convert all the text into lower letters
    # remove the words betweent brakets ()
    # remove these characters: {'$', ')', '?', '"', '’', '.',  '°', '!', ';', '/', "'", '€', '%', ':', ',', '('}
    # replace these spl characters with space: '\u200b', '\xa0', '-', '/'
    
    text = text.lower()
    text = decontractions(text)
    text = re.sub('[$)\?"’.°!;\'€%:,(/]', '', text)
    text = re.sub('\u200b', ' ', text)
    text = re.sub('\xa0', ' ', text)
    text = re.sub('-', ' ', text)
    return text


df['preprocessed_question'] = df['question'].apply(preprocess)
df['preprocessed_answer'] = df['answer'].apply(preprocess)
#df.head()







#https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html
#Setting Max_length to be 512 as discussed above
MAX_LENGTH = 512
import tensorflow as tf
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("vocab.txt")



# Tokenize, filter and pad sentences
def tokenize_and_filter(question, answer):
  tokenized_questions, tokenized_answers = [], []
  
  for (question, answer) in zip(question, answer):
    # generating sequences
    tokenized_question =  tokenizer.encode(question)
    tokenized_answer = tokenizer.encode(answer)
    
    tokenized_questions.append(tokenized_question)
    tokenized_answers.append(tokenized_answer)

  # padding the sequences
  tokenized_questions = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_questions, maxlen=MAX_LENGTH, padding='post')
  tokenized_answers = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_answers, maxlen=MAX_LENGTH, padding='post')
  
  return tokenized_questions, tokenized_answers

#tokenizing and padding the train questions and answers
q_list = df["preprocessed_question"].values.tolist()
a_list = df["preprocessed_answer"].values.tolist()

questions, answers = tokenize_and_filter(q_list, a_list)
#print(questions)


#preparing the question mask and the answer mask of the train dataset
train_question_mask=[[1 if token!=0 else 0 for token in question] for question in questions]
train_answer_mask=[[1 if token!=0 else 0 for token in answer] for answer in answers]

#print(train_answer_mask[9])


#creating the ffn layer 
#https://github.com/ash3n/DocProduct/blob/master/docproduct/models.py
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate,Conv1D,MaxPool1D,Dropout

class FFN(tf.keras.layers.Layer):
    def __init__(
            self,
            name='FFN',
            **kwargs):
        """Simple Dense wrapped with various layers
        """

        super(FFN, self).__init__(name=name, **kwargs)
        self.dropout = 0.2
        self.ffn_layer = tf.keras.layers.Dense(
            units=768,
            activation='relu',
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=32),name='FC1')
        

    def call(self, inputs):
        ffn_embedding = self.ffn_layer(inputs)
        ffn_embedding = tf.keras.layers.Dropout(
            self.dropout)(ffn_embedding)
        ffn_embedding += inputs
        return ffn_embedding
        
        
        
#creating the medicalbert model
#https://github.com/ash3n/DocProduct/blob/master/docproduct/models.py
from transformers import AutoTokenizer, TFAutoModel

#biobert_model = TFAutoModel.from_pretrained("cambridgeltl/BioRedditBERT-uncased")
#biobert_model.summary()

class MedicalQAModelwithBert(tf.keras.Model):
    def __init__(
            self,
            trainable=False,
            name=''):
        super(MedicalQAModelwithBert, self).__init__(name=name)

        self.q_ffn_layer = FFN(name='q_ffn')
        self.a_ffn_layer = FFN(name='a_ffn')
        self.biobert_model=biobert_model
        self.biobert_model.trainable=trainable
        self.cos=tf.keras.layers.Dot(axes=1,normalize=True)

    def call(self, inputs):
      question_embeddings=self.biobert_model(input_ids=inputs['question'],attention_mask=inputs['question_mask']).pooler_output
      answer_embeddings=self.biobert_model(input_ids=inputs['answer'],attention_mask=inputs['answer_mask']).pooler_output
      q_ffnn=self.q_ffn_layer(question_embeddings)
      a_ffnn=self.a_ffn_layer(answer_embeddings)
      output=self.cos([q_ffnn,a_ffnn])
      return {"label":output}
    
#res = FFN.call(df.ndim,df.shape)