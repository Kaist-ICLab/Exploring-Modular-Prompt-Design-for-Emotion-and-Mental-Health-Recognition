import pickle
import pandas as pd
from datasets import Dataset


def transform_labels(group_label, emotions):
	label_to_group = {}
	for group, labels in group_label.items():
		for label in labels:
			label_to_group[label] = group
	transformed_emotions = [label_to_group[emotion] for emotion in emotions]
	return transformed_emotions

label_info = {0: 'neutral', 1: 'joyful', 2: 'peaceful', 3: 'powerful', 4: 'scared', 5: 'mad', 6: 'sad'}
group_label = {0:['neutral'], 1: ['joyful','peaceful','powerful'], 2: ['scared','mad','sad']}
group_label_info = {0: 'positive', 1: 'negative', 2: 'neutral'}
with open('../data/emorynlp.pkl', 'rb') as f:
	data = pickle.load(f)
sentence_data = data[2]
label_data = data[1]

pandas_data = {'sentence':[], 'label':[],'label_text':[],'group_label': [], 'group_text': []}

for sentences,labels in zip(sentence_data,label_data):
	sentence = sentence_data[sentences]
	label = label_data[labels]
	pandas_data['sentence'].extend(sentence)
	pandas_data['label'].extend(label)
	pandas_data['label_text'].extend([label_info[text] for text in label])
	pandas_data['group_label'].extend(transform_labels(group_label,[label_info[text] for text in label]))
	pandas_data['group_text'].extend([group_label_info[text] for text in transform_labels(group_label,[label_info[text] for text in label])])

df = pd.DataFrame(pandas_data)
dataset = Dataset.from_pandas(df)
dataset.push_to_hub("KAIST-IC-LAB721/EmoryNLP-Classification")

### conversation ###
pandas_conversation = {'conversation':[], 'label':[],'label_text':[],'group_label': [], 'group_text': []}
for sentences,labels in zip(sentence_data,label_data):
	sentence = sentence_data[sentences]
	label = label_data[labels]
	pandas_conversation['conversation'].append(sentence)
	pandas_conversation['label'].append(label)
	pandas_conversation['label_text'].append([label_info[text] for text in label])
	pandas_conversation['group_label'].append(transform_labels(group_label, [label_info[text] for text in label]))
	pandas_conversation['group_text'].append([group_label_info[text] for text in transform_labels(group_label, [label_info[text] for text in label])])
df = pd.DataFrame(pandas_conversation)
dataset = Dataset.from_pandas(df)
dataset.push_to_hub("KAIST-IC-LAB721/EmoryNLP-Conversation")

print('debug')