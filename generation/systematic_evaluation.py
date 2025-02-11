import argparse
from dataset import load_dataset, preprocess_data_with_balanced_sampling
from prompt_generator import Prompt_Generator
from gpt import load_model, BaseModel
import random
from multiprocessing import Pool
import os


def gen(args, model_name, output_dir):
    dataset = load_dataset(dataset_name=args.data)
    dataset = preprocess_data_with_balanced_sampling(dataset_name=args.data, dataset=dataset, max_rows=args.max_rows)

    model = load_model(model_name, args)
    prompter = Prompt_Generator(args.data_task, args.problem_task, args.data, args.SI, args.TQ, args.PS, args.CT,
                                args.LD,args.OI)

    shot_count = 0
    shot_memory = ''

    if args.shot > 0:
        teacher_forcing = True
        base_model = BaseModel(api_key='', args=args)
        select_shot = random.sample(range(len(dataset['context'])), min(args.shot, len(dataset['context'])))
        for s in select_shot:
            context = dataset['context'][s]
            label = dataset['label'][s]
            label_text = dataset['label_text'][s]
            label_list = dataset['label_list'][s]

            if 'subject' in dataset and s < len(dataset['subject']):
                subject = dataset['subject'][s]
            else:
                subject = None

            shot = prompter(
                shot_memory='',
                context=context,
                label=label,
                label_text=label_text,
                label_list=label_list,
                subject=subject,
                shot_mode='few_shot',
                shot_count=shot_count
            )

            if teacher_forcing:
                system_prompt, user_prompt = base_model.split_prompt(shot, mode='few_shot')
                shot_memory += (system_prompt + user_prompt)
            else:
                sample = model.response(shot)
                shot_memory += sample[0] + sample[1]
            shot_count += 1

    param_dir = os.path.join(
        output_dir,
        args.data,
        args.problem_task,
        args.SI,
        args.TQ,
        f"PS-{args.PS}_shot-{args.shot}",
        model_name,
    )
    os.makedirs(param_dir, exist_ok=True)

    for count in range(len(dataset['context'])):
        if args.shot > 0 and count in select_shot:
            continue

        context = dataset['context'][count]
        label = dataset['label'][count]
        label_text = dataset['label_text'][count]
        label_list = dataset['label_list'][count]

        if 'subject' in dataset and count < len(dataset['subject']):
            subject = dataset['subject'][count]
        else:
            subject = None

        query = prompter(
            shot_memory=shot_memory,
            context=context,
            label=label,
            label_text=label_text,
            label_list=label_list,
            subject=subject,
            shot_count=shot_count
        )

        response = model.response(query)
        query_text, answer_text = response[0], response[1]

        with open(os.path.join(param_dir, f'query{count}.txt'), 'w', encoding='utf-8') as f:
            f.write(query_text)
        with open(os.path.join(param_dir, f'answer{count}.txt'), 'w', encoding='utf-8') as f:
            f.write(answer_text)
            if args.problem_task == 'Classification':
                f.write('\n\n' + 'TrueAnswer:' + str(dataset['label_text'][count]))
                f.write('\n\n' + 'TrueLabellist:' + str(dataset['label_list'][count]))

    print(f"Model {model_name} finished processing and results saved to {param_dir}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="systematic_evaluation Using SubProcess")
    parser.add_argument('--models', type=str, nargs='+', required=False, choices=['Gemini', 'Sonnet', 'GPT4o','Llama','Gemma','Qwen',
                                                                                  'Ollama_Llama','Ollama_Qwen','Ollama_Gemma', 'Ollama_Mistral','Ollama_Phi',
                                                                                  'Ollama_Qwen32B', 'OllamaPhi3_5'],
                        default=['Ollama_Llama'], help="Models to use for systematic evaluation")
    parser.add_argument('--data_task', type=str, required=False, choices=['Emotion', 'Mental-Health'],
                        default='Emotion', help="Data Task to use for systematic evaluation")
    parser.add_argument('--problem_task', type=str, required=False, choices=['Classification', 'Reasoning'],
                        default='Classification', help="Problem Task to use for systematic evaluation")
    parser.add_argument('--data', type=str, required=False,
                        choices=['iemocap', 'emobench', 'goemotion','dreaddit', 'cssrs', 'sdcnl'],
                        default='goemotion', help="Dataset to use for systematic evaluation")

    parser.add_argument('--SI', type=str, required=False,
                        default='persona-expert', help='System Instruction type')
    parser.add_argument('--TQ', type=str, required=False,
                        default='goemotion', help='Task Query type')
    parser.add_argument('--PS', type=str, required=False,
                        default='goemotion-none', help='Prompt strategy type')
    parser.add_argument('--CT', type=str, required=False,
                        default='goemotion', help='Context Input type')
    parser.add_argument('--LD', type=str, required=False,
                        default='none', help='Label Definition type')
    parser.add_argument('--OI', type=str, required=False,
                        default='goemotion', help='Output indicator type')
    parser.add_argument('--shot', type=int, required=False,
                        default=0, help='if shot > 0 few-shot else zero-shot')
    parser.add_argument('--max_rows', type=int, required=False, default=200, help='Maximum number of rows to load')
    parser.add_argument('--output_structure', type=str, required=False, choices=['index', 'newline'], default='index',
                        help="Structure of the output files: 'index' for indexed format, 'newline' for newline separated format")

    args = parser.parse_args()

    output_base_dir = '../results'
    os.makedirs(output_base_dir, exist_ok=True)

def main():
    with Pool(processes=len(args.models)) as pool:
        pool.starmap(gen, [(args, model, output_base_dir) for model in args.models])

if __name__ == '__main__':
    main()