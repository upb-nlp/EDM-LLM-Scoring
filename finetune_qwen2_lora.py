import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import wandb
import random
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from peft import LoraConfig
from prepare_datasets_final import prepare_selfexplanation_thinkaloud_se, prepare_selfexplanation_thinkaloud_ta, prepare_summary_aloe, prepare_paraphrasing_ulpc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login(key='ANONIMIZED')
wandb.init(
    project="LLM-Scoring",
    config={
        'task': 'Qwen2-7B, LoRA, all tasks',
    }
)

data_train_se, data_test_se, data_val_se = prepare_selfexplanation_thinkaloud_se()
data_train_ta, data_test_ta, data_val_ta = prepare_selfexplanation_thinkaloud_ta()
data_train_su, data_test_su, data_val_su = prepare_summary_aloe()
data_train_pa, data_test_pa, data_val_pa = prepare_paraphrasing_ulpc()

data_train = data_train_se + data_train_ta + data_train_su + data_train_pa
data_val = data_val_se + data_val_ta + data_val_su + data_val_pa

random.shuffle(data_train)
random.shuffle(data_val)

# Convert to dataset
train_dataset = Dataset.from_list(data_train)
val_dataset = Dataset.from_list(data_val)


model_name = "Qwen/Qwen2-7B"
new_model_name = "qwen2_7b_scoring_adapter_all_tasks"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token='ANONIMIZED', trust_remote_code=True)
tokenizer.pad_token_id = 151644

def tokenize_function(examples):
    inputs = tokenizer(f"{examples['prompt']}\n\n### Response: \n{examples['response_prompt']}", return_tensors="pt", max_length=2048, truncation=True)
    inputs["no_tokens"] = inputs["input_ids"].shape[1] + 1
    inputs["attention_mask"] = inputs["attention_mask"].squeeze()
    inputs["input_ids"] = inputs["input_ids"].squeeze()

    # Add tokenizer.bos_token_id to the beginning of the input_ids tokenizer.eos_token_id to the end of the input_ids. Update the attention mask
    inputs["input_ids"] = torch.cat((inputs["input_ids"], torch.tensor([tokenizer.eos_token_id])))
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.tensor([1])))

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

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token='ANONIMIZED', trust_remote_code=True)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset_tokenized,
    eval_dataset=val_dataset_tokenized,
    tokenizer=tokenizer,
    dataset_text_field='',
    args=TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=64,
        dataloader_num_workers=16,
        num_train_epochs=1,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=0.2,
        optim="adamw_8bit",
        report_to="wandb",
        output_dir=f"logs_{new_model_name}",
        save_strategy="no",
        save_steps=0.1,
        save_total_limit=1,
        fp16=True,
    ),
    max_seq_length=2000,
    peft_config=LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    ),
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False, return_tensors="pt", response_template="### Response:")
)

trainer.train()

# Save trained model
trainer.save_model(new_model_name)
tokenizer.save_pretrained(new_model_name)