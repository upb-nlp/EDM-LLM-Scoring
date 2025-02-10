import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm 
from prepare_datasets_final import prepare_summary_aloe, prepare_selfexplanation_thinkaloud_se, prepare_selfexplanation_thinkaloud_ta, prepare_paraphrasing_ulpc
import json
from peft import PeftModel

# ------------------------------------------------------------
TASK = 'paraphrasing' # summary selfexplanation thinkaloud paraphrasing
LORA = True # True False
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

# Generate scoring
access_token = 'ANONIMIZED'

if not LORA:
    model_name = "ANONIMIZED"
    filename = f"LLM-Scoring/experiments/{TASK}_generated_{model_name.split('/')[1]}.json"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=access_token)
    tokenizer.pad_token_id = 151644
    model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, device_map='auto')

else:
    base_model_name = "Qwen/Qwen2-7B"
    model_name = "ANONIMIZED"
    filename = f"LLM-Scoring/experiments/{TASK}_generated_{model_name.split('/')[1]}.json"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left", token=access_token)
    tokenizer.pad_token_id = 151644
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", token='ANONIMIZED')
    model = PeftModel.from_pretrained(base_model, model_name, device_map="auto", token='ANONIMIZED')
    model = model.merge_and_unload()

batch_size = 2
responses = []
for i in tqdm(range(0, len(data_test), batch_size)):
    end_interval = min(i+batch_size, len(data_test))

    texts = [f"{data['prompt']}\n\n### Response:" for data in data_test[i:end_interval]]

    model_inputs = tokenizer(texts, return_tensors="pt", padding='longest', truncation=True, max_length=2048*2).to(device)

    generated_ids = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=300,
        pad_token_id=151644,
        top_p=None,
        top_k=None,
        temperature=None,
        do_sample=False,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    responses += response

for data, response in zip(data_test, responses):
    data['generated_response'] = response

print(filename)
json.dump(data_test, open(filename, 'w'), indent=4)