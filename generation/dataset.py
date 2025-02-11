from collections import Counter
from sklearn.utils import resample
import datasets


def load_dataset(dataset_name: str) -> datasets.Dataset:
    huggingface_name = 'KAIST-IC-LAB721/'
    dataset = ''

    if dataset_name == "iemocap":
        dataset = 'IEMOCAP-Conversation'
    elif dataset_name == "emobench":
        dataset = 'EmoBench-eu'
    elif dataset_name == "dreaddit":
        dataset = 'Dreaddit'
    elif dataset_name == "cssrs":
        dataset = 'CSSRS-Suicide'
    elif dataset_name == "sdcnl":
        dataset = 'SDCNL'
    elif dataset_name == "goemotion":
        dataset = 'GoEmotion-Single'

    return datasets.load_dataset(huggingface_name + dataset)
def is_2d_list(lst):
    return all(isinstance(i, list) for i in lst)


def balanced_sampling(data, labels, max_rows):
    if is_2d_list(labels):
        labels = [item for sublist in labels for item in sublist]
    label_counts = Counter(labels)
    num_labels = len(label_counts)
    rows_per_label = max_rows // num_labels

    sampled_indices = []

    for label, count in label_counts.items():
        label_indices = [i for i, lbl in enumerate(labels) if lbl == label]

        if count >= rows_per_label:
            sampled_indices.extend(resample(label_indices, replace=False, n_samples=rows_per_label, random_state=42))
        else:
            sampled_indices.extend(resample(label_indices, replace=True, n_samples=rows_per_label, random_state=42))

    if len(sampled_indices) < max_rows:
        remaining_indices = list(set(range(len(labels))) - set(sampled_indices))
        additional_samples = resample(remaining_indices, replace=False, n_samples=max_rows - len(sampled_indices),
                                      random_state=42)
        sampled_indices.extend(additional_samples)

    return sampled_indices


def preprocess_data_with_balanced_sampling(dataset_name: str, dataset: datasets.Dataset, max_rows=200):
    if len(dataset['train']) < max_rows:
        max_rows = len(dataset['train'])

    data = {'context': [], 'label': [], 'label_text': [], 'conversations': [],
            'cause': [], 'cause_text': [], 'subject': [], 'label_list': []}

    if dataset_name == "iemocap":
        labels = dataset['train']['label']
        sampled_indices = balanced_sampling(dataset['train'], labels, max_rows)

        num_label_info = {0: 'happy', 1: 'sad', 2: 'neutral', 3: 'angry', 4: 'excited', 5: 'frustrated'}
        for idx in sampled_indices:
            data['context'].append({i: j for i, j in enumerate(dataset['train']['conversation'][idx])})
            data['label'].append({i: j for i, j in enumerate(dataset['train']['label'][idx])})
            data['label_text'].append({i: j for i, j in enumerate(dataset['train']['label_text'][idx])})
            data['label_list'].append(num_label_info)


    elif dataset_name == "emobench":
        dataset = dataset['train']
        data['context'].extend(dataset['scenario'])
        data['label'].extend(dataset['label'])
        data['label_text'].extend(dataset['label_text'])
        data['label_list'].extend(dataset['choices'])
        data['subject'].extend(dataset['subject'])

    elif dataset_name == "dreaddit":
        labels = dataset['train']['label']
        sampled_indices = balanced_sampling(dataset['train'], labels, max_rows)

        num_label_info = {0: 'yes', 1: 'no'}
        num_label_info = list(num_label_info.values())
        for idx in sampled_indices:
            data['context'].append(dataset['train']['post'][idx])
            data['label'].append(dataset['train']['label'][idx])
            data['label_text'].append(dataset['train']['label_text'][idx])

        data['label_list'].extend([num_label_info] * max_rows)

    elif dataset_name == "cssrs":
        labels = dataset['train']['label']
        sampled_indices = balanced_sampling(dataset['train'], labels, max_rows)

        num_label_info = {0: 'supportive', 1: 'indicator', 2: 'ideation', 3: 'behavior', 4: 'attempt'}
        num_label_info = list(num_label_info.values())
        for idx in sampled_indices:
            data['context'].append(dataset['train']['Post'][idx])
            data['label'].append(dataset['train']['label'][idx])
            data['label_text'].append(dataset['train']['label_text'][idx])
        data['label_list'].extend([num_label_info] * max_rows)
    elif dataset_name == "sdcnl":
        labels = dataset['train']['label']
        sampled_indices = balanced_sampling(dataset['train'], labels, max_rows)

        num_label_info = {0: 'depression', 1: 'suicidal'}
        num_label_info = list(num_label_info.values())

        for idx in sampled_indices:
            data['context'].append(dataset['train']['text'][idx])
            data['label'].append(dataset['train']['label'][idx])
            data['label_text'].append(dataset['train']['label_text'][idx])
        data['label_list'].extend([num_label_info] * max_rows)
    elif dataset_name == "goemotion":
        labels = dataset['train']['label']
        sampled_indices = balanced_sampling(dataset['train'], labels, max_rows)

        label_info = 'admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surpris, neutral'.replace(
            ' ', '')
        num_label_info = label_info.split(',')

        for idx in sampled_indices:
            data['context'].append(dataset['train']['sentence'][idx])
            data['label'].append(dataset['train']['label'][idx])
            data['label_text'].append(dataset['train']['label_text'][idx])
        data['label_list'].extend([num_label_info] * max_rows)

    return data
