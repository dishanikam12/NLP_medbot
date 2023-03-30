#for i in range(len(q_list)):
    #questions, answers = tokenize_and_filter(q_list[i], a_list[i])

#print(questions)


import pandas as pd
import numpy
#tokenized_questions, tokenized_answers = [], []


df = pd.read_json('biobert_data.json')
printdf.head()