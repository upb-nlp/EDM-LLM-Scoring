import json
import random
random.seed(42)

scoring_start_prompt = "Rate the quality of the following performed task, based on the scoring rubric."

def map_train_test_selfexplanation_thinkaloud_full(x):
    if x['Text'] in ['NS Text 1', 'RBC', '320']:
        return 'test'
    return 'train'

def compress_pa(x):
    if x == '0' or x == '1' or x == '2':
        return '0'
    if x == '3' or x == '4':
        return '1'
    if x == '5' or x == '6':
        return '2'

def get_task_rubric_prompt(scoring_details):
    task_prompt = scoring_details['task']
    scoring_rubric_prompt = ""
    for _, dict in scoring_details['scoring_rubric'].items():
        descriptions = '\n'.join([f"- - {dict['scores'][num]}: {dict['scores_description'][num]}" for num in dict['scores']])
        scoring_rubric_prompt += f"- {dict['name']}:\n{descriptions}\n"
    scoring_rubric_prompt = scoring_rubric_prompt[:-1]

    return task_prompt, scoring_rubric_prompt

def get_response_prompt(scoring_details, data):
    response_prompt = ""
    for rubric, dict in scoring_details['scoring_rubric'].items():
        response_prompt += f"- {dict['name']}: {dict['scores'][data[rubric]]}\n"
    response_prompt = response_prompt[:-1]
    return response_prompt

def prepare_selfexplanation_thinkaloud_se():
    dataset = json.load(open('LLM-Scoring/datasets_scoring_good/selfexplanation_thinkaloud_full.json', 'r'))
    dataset = [data for data in dataset if data['Proto'] == 'SE']
    scoring_details = json.load(open('LLM-Scoring/scoring_dicts/selfexplanation_thinkaloud_full_se.json', 'r'))
    task_prompt, scoring_rubric_prompt = get_task_rubric_prompt(scoring_details)

    for data in dataset:
        response_prompt = get_response_prompt(scoring_details, data)
        data['prompt'] = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Context: {data['context']}\n\n- Phrase: {data['TargetSentence']}\n\n### Execution: {data['ConstructedResponse']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"
        data['response_prompt'] = response_prompt

    for data in dataset:
        data['partition'] = map_train_test_selfexplanation_thinkaloud_full(data)

    dataset_train = [data for data in dataset if data['partition'] == 'train']
    dataset_dev = [data for data in dataset if data['partition'] == 'test']
    dataset_test = [data for data in dataset if data['partition'] == 'test']
    return dataset_train, dataset_test, dataset_dev

def prepare_selfexplanation_thinkaloud_ta():
    dataset = json.load(open('LLM-Scoring/datasets_scoring_good/selfexplanation_thinkaloud_full.json', 'r'))
    dataset = [data for data in dataset if data['Proto'] == 'TA']
    scoring_details = json.load(open('LLM-Scoring/scoring_dicts/selfexplanation_thinkaloud_full_ta.json', 'r'))
    task_prompt, scoring_rubric_prompt = get_task_rubric_prompt(scoring_details)

    for data in dataset:
        response_prompt = get_response_prompt(scoring_details, data)
        data['prompt'] = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Context: {data['context']}\n\n- Phrase: {data['TargetSentence']}\n\n### Execution: {data['ConstructedResponse']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"
        data['response_prompt'] = response_prompt

    for data in dataset:
        data['partition'] = map_train_test_selfexplanation_thinkaloud_full(data)

    dataset_train = [data for data in dataset if data['partition'] == 'train']
    dataset_dev = [data for data in dataset if data['partition'] == 'test']
    dataset_test = [data for data in dataset if data['partition'] == 'test']
    return dataset_train, dataset_test, dataset_dev

def prepare_summary_aloe():
    dataset = json.load(open('LLM-Scoring/datasets_scoring_good/summaries_aloe.json', 'r'))
    scoring_details = json.load(open('LLM-Scoring/scoring_dicts/summaries_aloe.json', 'r'))

    task_prompt, scoring_rubric_prompt = get_task_rubric_prompt(scoring_details)

    for data in dataset:
        if data['split'] == 'train' or data['split'] == 'val':
            for rubric in scoring_details['scoring_rubric_list']:
                if '.5' in data[rubric]:
                    # Randomly assign to one of the two closest integers
                    data[rubric] = str(random.choice([int(data[rubric].split('.')[0]), int(data[rubric].split('.')[0])+1]))
                else:
                    data[rubric] = str(int(float(data[rubric])))
            response_prompt = get_response_prompt(scoring_details, data)
            data['prompt'] = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Context: {data['context']}\n\n### Execution: {data['summary']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"
            data['response_prompt'] = response_prompt
        else:
            data['prompt'] = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Context: {data['context']}\n\n### Execution: {data['summary']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"


    dataset_train = [data for data in dataset if data['split'] == 'train']
    dataset_test = [data for data in dataset if data['split'] == 'test']
    dataset_val = [data for data in dataset if data['split'] == 'val']
    return dataset_train, dataset_test, dataset_val

def prepare_paraphrasing_ulpc():
    dataset = json.load(open('LLM-Scoring/datasets_scoring_good/paraphrasing_ulpc.json', 'r'))
    scoring_details = json.load(open('LLM-Scoring/scoring_dicts/paraphrasing_ulpc.json', 'r'))

    task_prompt, scoring_rubric_prompt = get_task_rubric_prompt(scoring_details)

    for data in dataset:
        for rubric in scoring_details['scoring_rubric_list']:
            data[rubric] = compress_pa(data[rubric])
        response_prompt = get_response_prompt(scoring_details, data)
        data['prompt'] = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Sentence: {data['Clean_Target_Sentence']}\n\n### Execution: {data['Clean_Utterance']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"
        data['response_prompt'] = response_prompt
    
    dataset_train = [data for data in dataset if data['trn_test_val'] == '1']
    dataset_test = [data for data in dataset if data['trn_test_val'] == '3']
    dataset_val = [data for data in dataset if data['trn_test_val'] == '2']
    return dataset_train, dataset_test, dataset_val

def prepare_essay_dress():
    dataset = json.load(open('LLM-Scoring/datasets_scoring_good/essay_dress.json', 'r'))
    scoring_details = json.load(open('LLM-Scoring/scoring_dicts/essay_dress.json', 'r'))

    _, scoring_rubric_prompt = get_task_rubric_prompt(scoring_details)

    for data in dataset:
        task_prompt = data['task_description']
        response_prompt = get_response_prompt(scoring_details, data)
        data['prompt'] = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n### Execution: {data['essay']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"
        data['response_prompt'] = response_prompt

    return None, dataset, None
