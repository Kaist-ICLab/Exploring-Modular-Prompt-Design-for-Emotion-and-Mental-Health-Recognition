import json

import pandas as pd
from datasets import Dataset

with open('../data/emobench/ea_data.json', encoding='UTF-8') as json_file:
    ea_data = json.load(json_file)
with open('../data/emobench/eu_data.json', encoding='UTF-8') as json_file:
    eu_data = json.load(json_file)

def find_label_index(label,choices):
    for i in range(len(choices)):
        if label[i] == choices:
            return i

pandas_ea_data = {'data_type':[], 'question_type':[],'relationship':[],'scenario': [], 'choices': [],'subject': [], 'score': [], 'label': [], 'label_str': []}


for data in ea_data:
    data_type = 'ea'
    question_type = data['Problem']
    relationship = data['Relationship']
    scenario = data['Scenario']['en']
    choices = data['Choices']['en']
    subject = data['Subject']['en']
    score = data['Score']
    label = data['Label']
    label_str = data['Label_str']['en']

    pandas_ea_data['data_type'].append(data_type)
    pandas_ea_data['question_type'].append(question_type)
    pandas_ea_data['relationship'].append(relationship)
    pandas_ea_data['scenario'].append(scenario)
    pandas_ea_data['subject'].append(subject)
    pandas_ea_data['choices'].append(choices)
    pandas_ea_data['score'].append(score)
    pandas_ea_data['label'].append(label)
    pandas_ea_data['label_str'].append(label_str)

df = pd.DataFrame(pandas_ea_data)
dataset = Dataset.from_pandas(df)
#dataset.push_to_hub("KAIST-IC-LAB721/EmoBench-ea")

pandas_eu_data = {'data_type':[], 'question_type':[],'scenario': [], 'choices': [],'subject': [],'label': [], 'label_text': [], 'cause_choices': [], 'cause_label': [],'cause_label_text': []}

for data in eu_data:
    data_type = 'eu'
    question_type = data['Category']
    scenario = data['Scenario']['en']
    choices = data['Emotion']['Choices']['en']
    subject = data['Subject']['en']
    label = data['Emotion']['Label']['en']
    cause_choices = data['Cause']['Choices']['en']
    cause_label = data['Cause']['Label']['en']

    pandas_eu_data['data_type'].append(data_type)
    pandas_eu_data['question_type'].append(question_type)
    pandas_eu_data['scenario'].append(scenario)
    pandas_eu_data['choices'].append(choices)
    pandas_eu_data['subject'].append(subject)
    pandas_eu_data['label_text'].append(label)
    pandas_eu_data['label'].append(find_label_index(choices,label))
    pandas_eu_data['cause_choices'].append(cause_choices)
    pandas_eu_data['cause_label'].append(find_label_index(cause_choices,cause_label))
    pandas_eu_data['cause_label_text'].append(cause_label)

df = pd.DataFrame(pandas_eu_data)
dataset = Dataset.from_pandas(df)
dataset.push_to_hub("KAIST-IC-LAB721/EmoBench-eu")


print('debug')