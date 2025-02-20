import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import wandb
import random
from trl import DataCollatorForCompletionOnlyLM
from prepare_datasets_final import prepare_selfexplanation_thinkaloud_se, prepare_selfexplanation_thinkaloud_ta, prepare_summary_aloe, prepare_paraphrasing_ulpc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TYPE = "summary" # paraphrasing, selfexplanation, thinkaloud, summary, all_tasks

wandb.login(key='TOKEN_KEY')
wandb.init(
    project="LLM-Scoring",
    config={
        'task': f"Llama 3.2 - 1B, full, {TYPE}",
    }
)

data_train_se, data_test_se, data_val_se = prepare_selfexplanation_thinkaloud_se()
data_train_ta, data_test_ta, data_val_ta = prepare_selfexplanation_thinkaloud_ta()
data_train_su, data_test_su, data_val_su = prepare_summary_aloe()
data_train_pa, data_test_pa, data_val_pa = prepare_paraphrasing_ulpc()

if TYPE == "selfexplanation":
    data_train = data_train_se
    data_val = data_val_se
elif TYPE == "thinkaloud":
    data_train = data_train_ta
    data_val = data_val_ta
elif TYPE == "summary":
    data_train = data_train_su
    data_val = data_val_su
elif TYPE == "paraphrasing":
    data_train = data_train_pa
    data_val = data_val_pa
elif TYPE == "all_tasks":
    data_train = data_train_se + data_train_ta + data_train_su + data_train_pa
    data_val = data_val_se + data_val_ta + data_val_su + data_val_pa

random.shuffle(data_train)
random.shuffle(data_val)

# Convert to dataset
train_dataset = Dataset.from_list(data_train)
val_dataset = Dataset.from_list(data_val)


model_name = "meta-llama/Llama-3.2-1B-Instruct"
new_model_name = f"llama32_1b_scoring_{TYPE}"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token='TOKEN_KEY', trust_remote_code=True)
tokenizer.pad_token_id = 128002

def tokenize_function(examples):
    messages = [
        {"role": "user", "content": examples['prompt']},
        {"role": "system", "content": examples['response_prompt']}
    ]

    text_example = tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False)

    inputs = tokenizer(text_example, return_tensors="pt", max_length=2048, truncation=True)
    inputs["no_tokens"] = inputs["input_ids"].shape[1] + 1
    inputs["attention_mask"] = inputs["attention_mask"].squeeze()
    inputs["input_ids"] = inputs["input_ids"].squeeze()

    # Add tokenizer.bos_token_id to the beginning of the input_ids tokenizer.eos_token_id to the end of the input_ids. Update the attention mask
    # inputs["input_ids"] = torch.cat((inputs["input_ids"], torch.tensor([tokenizer.eos_token_id])))
    # inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.tensor([1])))

    return inputs

train_dataset_tokenized = train_dataset.map(lambda x: tokenize_function(x))
val_dataset_tokenized = val_dataset.map(lambda x: tokenize_function(x))

print(len(train_dataset_tokenized))
print(len(val_dataset_tokenized))

# Filter out examples that are too long
train_dataset_tokenized = train_dataset_tokenized.filter(lambda x: x['no_tokens'] <= 2000)
val_dataset_tokenized = val_dataset_tokenized.filter(lambda x: x['no_tokens'] <= 2000)

print(len(train_dataset_tokenized))
print(len(val_dataset_tokenized))

# Drop all columns except input_ids, attention_mask and labels
train_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
val_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token='TOKEN_KEY', trust_remote_code=True)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset_tokenized,
    eval_dataset=val_dataset_tokenized,
    tokenizer=tokenizer,
    args=TrainingArguments(
        gradient_accumulation_steps=64,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        dataloader_num_workers=16,
        num_train_epochs=1,
        learning_rate=5e-6,
        lr_scheduler_type="constant",
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=0.2,
        optim="adamw_8bit",
        report_to="wandb",
        output_dir=f"logs_{new_model_name}",
        save_steps=0.2,
        save_total_limit=1,
    ),
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False, return_tensors="pt", response_template="<|start_header_id|>system<|end_header_id|>")
)

trainer.train()

# Save trained model
trainer.save_model(new_model_name)
tokenizer.save_pretrained(new_model_name)