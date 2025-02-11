import subprocess

model_parameters = ['GPT4o']

# Dataset and parameters setting
tasks = [
    {
        'data_task': 'Emotion',
        'problem_task': 'Classification',
        'data': 'emobench',
        'TQ': ['emobench-Clear', 'emobench-EmDe', 'emobench-Ana'],
        'CT': 'emobench',
        'OI': 'emobench',
        'SI': 'persona-none',
        'PS_base': 'emobench'
    },
    {
        'data_task': 'Emotion',
        'problem_task': 'Classification',
        'data': 'goemotion',
        'TQ': ['goemotion-Clear', 'goemotion-EmDe', 'goemotion-Ana'],
        'CT': 'goemotion',
        'OI': 'goemotion',
        'SI': 'persona-none',
        'PS_base': 'goemotion'
    },
    {
        'data_task': 'Mental-Health',
        'problem_task': 'Classification',
        'data': 'dreaddit',
        'TQ': ['dreaddit-Clear', 'dreaddit-EmDe', 'dreaddit-Ana'],
        'CT': 'dreaddit',
        'OI': 'dreaddit',
        'SI': 'persona-none',
        'PS_base': 'dreaddit'
    },
    {
        'data_task': 'Mental-Health',
        'problem_task': 'Classification',
        'data': 'cssrs',
        'TQ': ['cssrs-Clear', 'cssrs-EmDe', 'cssrs-Ana'],
        'CT': 'cssrs',
        'OI': 'cssrs',
        'SI': 'persona-none',
        'PS_base': 'cssrs'
    },
    {
        'data_task': 'Mental-Health',
        'problem_task': 'Classification',
        'data': 'sdcnl',
        'TQ': ['sdcnl-Clear', 'sdcnl-EmDe', 'sdcnl-Ana'],
        'CT': 'sdcnl',
        'OI': 'sdcnl',
        'SI': 'persona-none',
        'PS_base': 'sdcnl'
    }
]


def generate_commands(max_rows=200):
    commands = []
    for task in tasks:
        for TQ in task['TQ']:
            for shot in range(0,1):  # shot 0~3
                # When the shot is 0, set PS to '{dataset}-none'; when the shot is 1 or more, set it to '{dataset}-fewshot_icl'.
                PS = f"{task['PS_base']}-none" if shot == 0 else f"{task['PS_base']}-fewshot_icl"

                commands.append([
                    '--data_task', task['data_task'],
                    '--problem_task', task['problem_task'],
                    '--data', task['data'],
                    '--SI', task['SI'],
                    '--TQ', TQ,
                    '--PS', PS,
                    '--CT', task['CT'],
                    '--LD', 'none',
                    '--OI', task['OI'],
                    '--shot', str(shot),
                    '--max_rows', str(max_rows),
                    '--output_structure', 'index'
                ])
    return commands


def run_command_for_model(model, commands):
    for command in commands:
        full_command = ['python', 'systematic_evaluation.py', '--models', model] + command
        print(f"Running command for model {model}: {' '.join(full_command)}")

        try:
            result = subprocess.run(full_command, check=True, text=True, capture_output=True)

            print(f"Standard Output for {model}:\n{result.stdout}")
            print(f"Standard Error for {model}:\n{result.stderr}")

        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running model {model}.\nError Message: {e.stderr}")
            return e.returncode

    return 0


def run_all_models():
    commands = generate_commands(max_rows=200)
    for model in model_parameters:
        result = run_command_for_model(model, commands)
        if result != 0:
            break


if __name__ == "__main__":
    run_all_models()
