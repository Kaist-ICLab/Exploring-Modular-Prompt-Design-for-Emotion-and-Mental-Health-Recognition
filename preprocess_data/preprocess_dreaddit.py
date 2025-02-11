from datasets import load_dataset, concatenate_datasets

def map_label_text_to_label(example):
    example['label'] = 0 if example['label_text'] == 'yes' else 1
    return example

dt = load_dataset("asmaab/dreadditTraining")
dv = load_dataset("asmaab/DreadditValidation")

train_dataset = dt['train']
validation_dataset = dv['validation']
dataset_df = concatenate_datasets([train_dataset, validation_dataset])
dataset_df = dataset_df.rename_column("label", "label_text")
dataset_df = dataset_df.rename_column("question", "example_question")


# Apply the mapping function to the dataset
dataset_df = dataset_df.map(map_label_text_to_label)
dataset_df.push_to_hub("KAIST-IC-LAB721/Dreaddit")
print('debug')
