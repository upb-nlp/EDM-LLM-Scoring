import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm 
from prepare_datasets_final import prepare_summary_aloe, prepare_selfexplanation_thinkaloud_se, prepare_selfexplanation_thinkaloud_ta, prepare_paraphrasing_ulpc, prepare_essay_dress
import json
from peft import PeftModel

# ------------------------------------------------------------
TASK = 'paraphrasing' # summary selfexplanation thinkaloud paraphrasing
LORA = False # True False
# ------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if TASK == 'summary':
    _, data_test, _ = prepare_summary_aloe()
elif TASK == 'selfexplanation':
    _, data_test, _ = prepare_selfexplanation_thinkaloud_se()
elif TASK == 'thinkaloud':
    _, data_test, _ = prepare_selfexplanation_thinkaloud_ta()
elif TASK == 'paraphrasing':
    _, data_test, _ = prepare_paraphrasing_ulpc()
elif TASK == 'essay':
    _, data_test, _ = prepare_essay_dress()

# Generate scoring
access_token = 'TOKEN_KEY'

if not LORA:
    model_name = "flant5_1b_scoring_paraphrasing"
    filename = f"LLM-Scoring/experiments/{TASK}_generated_{model_name.split('/')[0]}.json"

    tokenizer = T5Tokenizer.from_pretrained(model_name, token=access_token)
    model = T5ForConditionalGeneration.from_pretrained(model_name, token=access_token, device_map='auto')

batch_size = 2
responses = []
for i in tqdm(range(0, len(data_test), batch_size)):
    end_interval = min(i+batch_size, len(data_test))

    texts = [f"{data['prompt']}" for data in data_test[i:end_interval]]

    model_inputs = tokenizer(texts, return_tensors="pt", padding='longest', truncation=True, max_length=2048*2).to(device)

    generated_ids = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=3000,
        top_p=None,
        top_k=None,
        temperature=None,
        do_sample=False,
    )

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    responses += response

for data, response in zip(data_test, responses):
    data['generated_response'] = response

print(filename)
json.dump(data_test, open(filename, 'w'), indent=4)