import json
import pandas as pd
from datasets import Dataset, load_dataset
def main():
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    data = ds['test']
    label_info = 'admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surpris, neutral'.replace(' ','')
    label_info = label_info.split(',')

    label_info = {v:k for k,v in enumerate(label_info)}
    reversed_label_info = {v: k for k, v in label_info.items()}

    pandas_data = {'sentence':[], 'label':[],'label_text':[]}


    for i in range(len(data['text'])):
        #if len(data[i]['labels']) == 1:
        # pandas_data['sentence'].append(
        #     data[i]['text']
        # )

        if len(data[i]['labels']) == 1:
            # label_list = []
            # label = data[i]['labels'][0]
            # label_list.append(reversed_label_info[label])
            # pandas_data['label_text'].append(
            #     label_list
            # )
            # label_list = [data[i]['labels'][0]]
            # pandas_data['label'].append(
            #     label_list
            # )
            continue
        else:
            pandas_data['sentence'].append(
                data[i]['text']
            )
            label_list = []
            for j in range(len(data[i]['labels'])):
                label = data[i]['labels'][j]
                label_list.append(reversed_label_info[int(label)])
            pandas_data['label_text'].append(label_list)

            label_list = []
            for j in range(len(data[i]['labels'])):
                label_list.append(
                    data[i]['labels'][j]
                )
            pandas_data['label'].append(
                label_list
            )

    df = pd.DataFrame(pandas_data)
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub("KAIST-IC-LAB721/GoEmotion-Multiple")

    print('debug')
if __name__ == '__main__':
    main()