import pandas as pd
from datasets import Dataset

file = '../data/CDSNL/combined-set.csv'
df=pd.read_csv(file)

label_info = {0: 'depression',1: 'suicidal'}

title = list(df['title'])
text = list(df['selftext'])
label = list(df['is_suicide'])
label_text = [label_info[i] for i in label]
data = {'title': title, 'text': text, 'label': label, 'label_text': label_text}

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)
dataset.push_to_hub("KAIST-IC-LAB721/CDSNL")
print('debug')
