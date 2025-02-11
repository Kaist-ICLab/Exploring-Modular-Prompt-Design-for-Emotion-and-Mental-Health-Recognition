import json
import os
from string import Formatter
import ast

class SafeFormatter(Formatter):
    def get_value(self, key, args, kwargs):
        return kwargs.get(key, '')


def conditional_format(template, **kwargs):
    formatter = SafeFormatter()

    # Extract all variables used in the template
    used_vars = [fname for _, fname, _, _ in formatter.parse(template) if fname]

    # Filter the kwargs to only include the variables used in the template
    relevant_kwargs = {key: value for key, value in kwargs.items() if key in used_vars and value}

    return formatter.format(template, **relevant_kwargs)


class Prompt_Generator():
    def __init__(self, data_task, problem_task, data_name, SI, TQ, PS, CT, LD, OI, shot=0):
        self.template = self.load_prompt_template(data_task, problem_task)
        self.data_task = data_task
        self.problem_task = problem_task
        self.data_name = data_name
        self.SI = SI
        self.TQ = TQ
        self.PS = PS
        self.CT = CT
        self.LD = LD
        self.OI = OI
        self.shot = shot
        self.prompt_template = self.extract_prompt_template()

    def __call__(self, shot_memory='', context='', label='', label_text='', label_list='', subject='', shot_mode='basic', shot_count=0):
        shot = conditional_format(self.prompt_template['few_shot'], shot_memory=shot_memory)
        system_instruction = conditional_format(self.prompt_template['system_instruction'], context=context, subject=subject, label_list=label_list, label_text=label_text)
        task_query = conditional_format(self.prompt_template['task_query'], context=context, subject=subject, label_list=label_list, label_text=label_text)
        prompt_strategy = conditional_format(self.prompt_template['prompt_strategy'], context=context, subject=subject, label_list=label_list, label_text=label_text)
        label_def = ast.literal_eval(self.prompt_template['label_def'].replace('‘', "'").replace('’', "'"))
        label_list_for_label_def = [word.strip() for item in label_list for word in item.split('&')]
        label_def_context = {i: label_def[i] for i in label_list_for_label_def if i in label_def}
        if len(label_def_context.keys()) == 0:
            label_def_context = ''
        output_indicator = conditional_format(self.prompt_template['output_indicator'], context=context, subject=subject, label_list=label_list, label_text=label_text)

        if shot_mode == 'few_shot':
            system_instruction = ''
            task_query = ''
            prompt_strategy = ''
            output_indicator = ''
            context = conditional_format(self.prompt_template['prompt_strategy'], context=context, subject=subject,
                                         label_list=label_list, label_text=label_text)
        else:
            if shot_count != 0:
                prompt_strategy = ''
            context = conditional_format(self.prompt_template['context'], context=context, subject=subject,
                                         label_list=label_list, label_text=label_text)

        # if shot_count > 0:
        #     system_instruction = ''
        #     task_query = ''
        #
        # if shot_count == 0 and shot_mode == 'basic':
        #     # Basic 모드일 때만 system_instruction과 task_query를 유지
        #     prompt_strategy = self.prompt_template['prompt_strategy']
        #     system_instruction = self.prompt_template['system_instruction']
        #     task_query = self.prompt_template['task_query']

        prompt = {
            'few_shot': shot,
            'system_instruction': system_instruction,
            'task_query': task_query,
            'prompt_strategy': prompt_strategy,
            'context': context,
            'label_def': label_def_context,
            'output_indicator': output_indicator
        }

        return prompt

    def load_prompt_template(self, data_task, problem_task):
        templates = {}
        directory = f"../prompt_template/{data_task}/{problem_task}/"
        if os.path.exists(directory) and os.path.isdir(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.json'):
                    filepath = os.path.join(directory, filename)
                    with open(filepath, 'r', encoding='utf-8') as file:
                        try:
                            templates[filename] = json.load(file)
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON from file {filepath}: {e}")
        else:
            print(f"Directory {directory} does not exist.")
        return templates

    def extract_prompt_template(self):
        prompt = {
            'few_shot': '{shot_memory}',
            'system_instruction': '',
            'task_query': '',
            'prompt_strategy': '',
            'context': '{context}',
            'label_def': '',
            'output_indicator': ''
        }

        system_instruction_type = self.SI.split('-')
        system_instruction_template = self.template['System_Instruction.json']
        for key in system_instruction_type:
            system_instruction_template = system_instruction_template[key]
        prompt['system_instruction'] = system_instruction_template

        task_query_type = self.TQ.split('-')
        task_query_template = self.template['Task_Query.json']
        for key in task_query_type:
            task_query_template = task_query_template[key]
        prompt['task_query'] = task_query_template

        prompt_strategy_type = self.PS.split('-')
        prompt_strategy_template = self.template['Prompt_Strategy.json']
        for key in prompt_strategy_type:
            prompt_strategy_template = prompt_strategy_template[key]
        prompt['prompt_strategy'] = prompt_strategy_template

        context_type = self.CT.split('-')
        context_template = self.template['Context_Input.json']
        for key in context_type:
            context_template = context_template[key]
        prompt['context'] = context_template

        label_def_type = self.LD.split('-')
        label_def_template = self.template['Label_Def.json']
        for key in label_def_type:
            label_def_template = label_def_template[key]
        prompt['label_def'] = label_def_template

        output_indicator_type = self.OI.split('-')
        output_indicator_template = self.template['Output_Indicator.json']
        for key in output_indicator_type:
            output_indicator_template = output_indicator_template[key]
        prompt['output_indicator'] = output_indicator_template

        return prompt

    def gen(self):
        pass
