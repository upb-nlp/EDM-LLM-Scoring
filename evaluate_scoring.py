import json
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

TASK = 'summary' # summary selfexplanation thinkaloud paraphrasing
ALL_TASKS = 'all_tasks'
MODEL = TASK # TASK ALL_TASKS

def extract_score_from_response(response):
    # Split the response into lines
    lines = response.split('\n')

    scores = {}
    for line in lines:
        if line.startswith('- '):
            # Remove <|endoftext|> from the line if it exists
            line = line.replace('<|endoftext|>', '')
            line = line.replace('<|im_start|>', '')
                            
            try:
                key, value = line.split(': ')
                if key[2:] not in scores:
                    scores[key[2:]] = value
            except:
                pass
    return scores

def _get_response_prompt(scoring_details, data):
    response_prompt = ""
    for rubric, dict in scoring_details['scoring_rubric'].items():
        response_prompt += f"- {dict['name']}: {dict['scores'][data[rubric]]}\n"
    response_prompt = response_prompt[:-1]
    return response_prompt

def _get_summary_score(scoring_details_rubric_scores_dict, initial_score, generated_response_rubric):
    for key, value in scoring_details_rubric_scores_dict.items():
        if value == generated_response_rubric:
            generated_score = key
            break
    
    initial_score = float(initial_score)
    generated_score = float(generated_score)

    if int(initial_score) == int(generated_score) or int(initial_score) + 1 == int(generated_score):
        return str(int(generated_score))
    elif abs(int(initial_score) - int(generated_score)) < abs(int(initial_score) + 1 - int(generated_score)):
        return str(int(initial_score))
    else:
        return str(int(initial_score) + 1)

def add_response_prompt_summary(dataset, scoring_details):
    for data in dataset:
        generated_response_dict = extract_score_from_response(data['generated_response'])
        for rubric in scoring_details['scoring_rubric_list']:
            if '.5' in data[rubric]:
                data[rubric] = _get_summary_score(scoring_details['scoring_rubric'][rubric]['scores'], data[rubric], generated_response_dict[scoring_details['scoring_rubric'][rubric]['name']])
            else:
                data[rubric] = str(int(float(data[rubric])))
        response_prompt = _get_response_prompt(scoring_details, data)
        data['response_prompt'] = response_prompt

data_test = json.load(open(f"LLM-Scoring/experiments/{TASK}_generated_qwen2_7b_scoring_adapter_{MODEL}.json", 'r'))

if TASK == 'selfexplanation':
    scoring_dict = json.load(open(f"LLM-Scoring/scoring_dicts/selfexplanation_thinkaloud_full_se.json", 'r'))
elif TASK == 'thinkaloud':
    scoring_dict = json.load(open(f"LLM-Scoring/scoring_dicts/selfexplanation_thinkaloud_full_ta.json", 'r'))
elif TASK == 'summary':
    scoring_dict = json.load(open(f"LLM-Scoring/scoring_dicts/summaries_aloe.json", 'r'))
    add_response_prompt_summary(data_test, scoring_dict)
elif TASK == 'paraphrasing':
    scoring_dict = json.load(open(f"LLM-Scoring/scoring_dicts/paraphrasing_ulpc.json", 'r'))

scoring_rubrics = [(k, sc['name']) for k, sc in scoring_dict['scoring_rubric'].items()]
# Extract scoring
for data in data_test:
    data['actual_scores'] = extract_score_from_response(data['response_prompt'])
    data['generated_scores'] = extract_score_from_response(data['generated_response'])


accuracy_scores = {}
f1_scores = {}
mean_error_scores = {}

for sc in scoring_rubrics:
    scoring_rubric = sc[1]
    k = sc[0]
    y_true = [data['actual_scores'][scoring_rubric] for data in data_test]
    y_pred = [data['generated_scores'][scoring_rubric] if scoring_rubric in data['generated_scores'] else '' for data in data_test]

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    accuracy_scores[scoring_rubric] = accuracy

    # Calculate f1
    f1 = f1_score(y_true, y_pred, average='weighted')
    f1_scores[scoring_rubric] = f1


    # Calculate mean error
    correspondence = {value: int(key) for key, value in scoring_dict['scoring_rubric'][k]['scores'].items() if key.lower() != 'blank'}
    y_true_numeric = [correspondence[label] if label in correspondence else -1 for label in y_true]
    y_pred_numeric = [correspondence[label] if label in correspondence else -1 for label in y_pred]

    mean_error = sum([abs(y_true_numeric[i] - y_pred_numeric[i]) for i in range(len(y_true_numeric))]) / len(y_true_numeric)
    mean_error_scores[scoring_rubric] = mean_error

print(f"--------------- {TASK} ---------------")
print(f"--------------- {MODEL} ---------------")
#print(f"Accuracy scores: {accuracy_scores}")
#print(f"F1 scores: {f1_scores}")
#print(f"Mean error scores: {mean_error_scores}")
print()
# Print these scores as a table
print("Scoring rubric | Accuracy | F1 | Mean error")
print("-------------- | -------- | -- | ----------")
for sc in scoring_rubrics:
    scoring_rubric = sc[1]
    print(f"{scoring_rubric} | {accuracy_scores[scoring_rubric]:.2f} | {f1_scores[scoring_rubric]:.2f} | {mean_error_scores[scoring_rubric]:.2f}")
