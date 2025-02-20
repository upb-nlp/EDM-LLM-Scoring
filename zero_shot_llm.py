import json
from prepare_datasets_final import prepare_selfexplanation_thinkaloud_se, prepare_selfexplanation_thinkaloud_ta, prepare_summary_aloe, prepare_paraphrasing_ulpc
from ollama import chat, ChatResponse, Client
from tqdm import tqdm

data_train_se, data_test_se, data_val_se = prepare_selfexplanation_thinkaloud_se()
data_train_ta, data_test_ta, data_val_ta = prepare_selfexplanation_thinkaloud_ta()
data_train_su, data_test_su, data_val_su = prepare_summary_aloe()
data_train_pa, data_test_pa, data_val_pa = prepare_paraphrasing_ulpc()

client = Client()

TASK = 'thinkaloud' # 'selfexplanation', 'thinkaloud', 'summary', 'paraphrasing'

if TASK == 'selfexplanation':
    data_test = data_test_se
elif TASK == 'thinkaloud':
    data_test = data_test_ta
elif TASK == 'summary':
    data_test = data_test_su
elif TASK == 'paraphrasing':
    data_test = data_test_pa

with open(f"{TASK}_generated_few_shot.jsonl", "wt") as f:
    for data in tqdm(data_test):
        prompt = data['prompt'] + "\n\nPlease return a single JSON with the following format: { {{Scoring rubric}}: {{score}} }"
        response = client.generate(model="llama3.3:latest", prompt=prompt, system="You are an educational expert.", options={"num_ctx": 8128, "temperature": 0}, format="json")
        try:
            response = json.loads(response["response"])
            data['scores'] = response
        except:
            data['scores'] = None
        f.write(json.dumps(data) + "\n")
