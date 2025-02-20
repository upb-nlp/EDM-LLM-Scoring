import torch
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import wandb
import random
from prepare_datasets_final import prepare_selfexplanation_thinkaloud_se, prepare_selfexplanation_thinkaloud_ta, prepare_summary_aloe, prepare_paraphrasing_ulpc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login(key='TOKEN_KEY')
wandb.init(
    project="LLM-Scoring",
    config={
        'task': 'FlanT5-l, full, all_tasks',
    }
)

data_train_se, data_test_se, data_val_se = prepare_selfexplanation_thinkaloud_se()
data_train_ta, data_test_ta, data_val_ta = prepare_selfexplanation_thinkaloud_ta()
data_train_su, data_test_su, data_val_su = prepare_summary_aloe()
data_train_pa, data_test_pa, data_val_pa = prepare_paraphrasing_ulpc()

data_train = data_train_pa + data_train_se + data_train_ta + data_train_su
data_val = data_val_pa + data_val_se + data_val_ta + data_val_su

random.shuffle(data_train)
random.shuffle(data_val)

# Convert to dataset
train_dataset = Dataset.from_list(data_train)
val_dataset = Dataset.from_list(data_val)


model_name = "google/flan-t5-large"
new_model_name = "flant5_1b_scoring_all_tasks"

tokenizer = T5Tokenizer.from_pretrained(model_name, token='TOKEN_KEY', trust_remote_code=True)

def tokenize_function(examples):
    inputs = tokenizer(f"{examples['prompt']}", return_tensors="pt", max_length=2048, truncation=True)
    inputs["attention_mask"] = inputs["attention_mask"].squeeze()
    inputs["input_ids"] = inputs["input_ids"].squeeze()
    inputs["labels"] = tokenizer(f"{examples['response_prompt']}", return_tensors="pt", max_length=2048, truncation=True)["input_ids"].squeeze()

    # Add tokenizer.eos_token_id to the end of the labels.
    # inputs["labels"] = torch.cat((inputs["labels"], torch.tensor([tokenizer.eos_token_id])))
    inputs["no_tokens"] = inputs["input_ids"].shape[0] + inputs["labels"].shape[0]

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
train_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", token='TOKEN_KEY', trust_remote_code=True)

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
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=0.5,
        optim="adamw_8bit",
        report_to="wandb",
        output_dir=f"logs_{new_model_name}",
        save_steps=0.2,
        save_total_limit=1,
    ),
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="pt")
)

trainer.train()

# Save trained model
trainer.save_model(new_model_name)
tokenizer.save_pretrained(new_model_name)