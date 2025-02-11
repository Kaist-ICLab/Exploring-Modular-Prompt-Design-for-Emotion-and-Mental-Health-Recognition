import pandas as pd
from datasets import Dataset

label_rate = {0:'Supportive', 1: 'Indicator', 2:'Ideation', 3:'Behavior', 4:'Attempt'}
label_rate_reverse = {v:k for k,v in label_rate.items()}
file = '../data/CSSR/500_anonymized_Reddit_users_posts_labels - 500_anonymized_Reddit_users_posts_labels.csv'
df=pd.read_csv(file)
post = list(df['Post'])
clear_post = [i[2:-2] for i in post]
label = list(df['Label'])
data = {'Post': clear_post, 'label': [label_rate_reverse[lbl] for lbl in list(df['Label'])], 'label_text':list(df['Label'])}

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)
dataset.push_to_hub("KAIST-IC-LAB721/CSSRS-Suicide")
print('debug')
