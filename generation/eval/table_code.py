import os
import argparse
import pandas as pd

def extract_metrics_from_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()

    metrics = []
    current_metrics = {
        'Base Path': None,
        'SI': None,
        'TQ': None,
        'PS': "none",
        'Shot': None,
        'Model': None,
        'F1-macro': None,
        'F1-weight': None,
        'Precision-macro': None,
        'Recall-macro': None,
        'Accuracy': None,
        'Avg Confidence Score': None,
        'Correlation between Confidence Score and Accuracy': None,
        'Processed files': None,
        'Failed files': None,
        'Error processing files': None,
        'Classification Report': None
    }

    path_parts = file_path.split(os.sep)

    try:
        current_metrics['Base Path'] = os.sep.join(path_parts[:6])

        if 'persona-expert' in path_parts:
            current_metrics['SI'] = 'expert'
        elif 'persona-none' in path_parts:
            current_metrics['SI'] = 'none'

        for part in path_parts:
            if 'Ana' in part:
                current_metrics['TQ'] = 'Ana'
            elif 'EmDe' in part:
                current_metrics['TQ'] = 'EmDe'
            elif 'Clear' in part:
                current_metrics['TQ'] = 'Clear'

        if 'ICL' in path_parts:
            current_metrics['PS'] = 'ICL'

        for part in path_parts:
            if 'shot-' in part:
                current_metrics['Shot'] = part.split('-')[-1]

    except IndexError as e:
        print(f"Error processing file path {file_path}: {e}")

    report_start = False
    report_lines = []
    for line in data:
        if line.strip().startswith('# Model:'):
            if current_metrics['Model'] is not None:
                if report_lines:
                    current_metrics['Classification Report'] = "\n".join(report_lines)
                    report_lines = []
                metrics.append(current_metrics.copy())
            current_metrics['Model'] = line.split(':', 1)[1].strip()
        elif '**F1-macro:**' in line:
            current_metrics['F1-macro'] = float(line.split('**F1-macro:**')[1].strip())
        elif '**F1-weight:**' in line:
            current_metrics['F1-weight'] = float(line.split('**F1-weight:**')[1].strip())
        elif '**Precision-macro:**' in line:
            current_metrics['Precision-macro'] = float(line.split('**Precision-macro:**')[1].strip())
        elif '**Recall-macro:**' in line:
            current_metrics['Recall-macro'] = float(line.split('**Recall-macro:**')[1].strip())
        elif '**Accuracy:**' in line:
            current_metrics['Accuracy'] = float(line.split('**Accuracy:**')[1].strip())
        elif '**Avg Confidence Score:**' in line:
            current_metrics['Avg Confidence Score'] = float(line.split('**Avg Confidence Score:**')[1].strip())
        elif '**Correlation between Confidence Score and Accuracy:**' in line:
            current_metrics['Correlation between Confidence Score and Accuracy'] = float(
                line.split('**Correlation between Confidence Score and Accuracy:**')[1].strip())
        elif '**Processed files:**' in line:
            current_metrics['Processed files'] = int(line.split('**Processed files:**')[1].strip())
        elif '**Failed files:**' in line:
            try:
                current_metrics['Failed files'] = int(float(line.split('**Failed files:**')[1].strip()))
            except ValueError:
                current_metrics['Failed files'] = line.split('**Failed files:**')[1].strip()
        elif '**Error processing files:**' in line:
            current_metrics['Error processing files'] = line.split('**Error processing files:**')[1].strip()
        elif line.strip() == '## Classification Report':
            report_start = True
        elif report_start:
            if line.strip() == '---':
                report_start = False
            else:
                report_lines.append(line.rstrip())

    if current_metrics['Model'] is not None:
        if report_lines:
            current_metrics['Classification Report'] = "\n".join(report_lines)
        metrics.append(current_metrics)

    return metrics

def traverse_and_collect(folder_path):
    collected_metrics = []
    dataset_name = os.path.basename(folder_path)

    for entry in os.scandir(folder_path):
        if entry.is_dir():
            sub_metrics, _ = traverse_and_collect(entry.path)
            collected_metrics.extend(sub_metrics)
        elif entry.is_file() and entry.name.endswith('.md'):
            metrics = extract_metrics_from_md(entry.path)
            collected_metrics.extend(metrics)

    return collected_metrics, dataset_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process result files and calculate metrics.')
    parser.add_argument('--dataset', type=str, required=False,
                        default='emobench/Classification',
                        help='Path to the folder containing result files relative to the base folder')
    parser.add_argument('--base_folder_path', type=str, required=False,
                        default='../../results/',
                        help='Base path to the folder containing result files')

    args = parser.parse_args()

    base_folder_path = os.path.join(args.base_folder_path, args.dataset)

    metrics_list, dataset_name = traverse_and_collect(base_folder_path)

    df = pd.DataFrame(metrics_list)
    output_dir = 'table_result'
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, f'{args.dataset[:-15]}_results.csv')

    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Results saved to {output_csv}")
