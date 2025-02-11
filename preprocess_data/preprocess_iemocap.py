import pickle
import pandas as pd
from datasets import Dataset

label_info = {0: 'happy', 1: 'sad', 2: 'neutral', 3: 'angry', 4: 'excited', 5: 'frustrated'}

with open('../data/iemocap.pkl', 'rb') as f:
	data = pickle.load(f)
sentence_data = data[2]
label_data = data[1]

pandas_data = {'sentence':[], 'label':[],'label_text':[]}

for sentences,labels in zip(sentence_data,label_data):
	sentence = sentence_data[sentences]
	label = label_data[labels]
	pandas_data['sentence'].extend(sentence)
	pandas_data['label'].extend(label)
	pandas_data['label_text'].extend([label_info[text] for text in label])

df = pd.DataFrame(pandas_data)
dataset = Dataset.from_pandas(df)
dataset.push_to_hub("KAIST-IC-LAB721/IEMOCAP-Classification")

### conversation ###
pandas_conversation = {'conversation':[],'label':[], 'label_text':[]}
for sentences,labels in zip(sentence_data,label_data):
	sentence = sentence_data[sentences]
	label = label_data[labels]
	pandas_conversation['conversation'].append(sentence)
	pandas_conversation['label'].append(label)
	pandas_conversation['label_text'].append([label_info[text] for text in label])

df = pd.DataFrame(pandas_conversation)
dataset = Dataset.from_pandas(df)
dataset.push_to_hub("KAIST-IC-LAB721/IEMOCAP-Conversation")

print('debug')