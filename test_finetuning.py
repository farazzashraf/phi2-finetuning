from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load your custom JSONL dataset
dataset = load_dataset("json", data_files="data\coco_dataset_bi.jsonl", split="train")
dataset = dataset.select(range(10000))
dataset = dataset.train_test_split(test_size=0.1)
train_data = dataset["train"]
val_data = dataset["test"]

# ✅ Format instruction-style text
def format_instruction(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    }

formatted_train = train_data.map(format_instruction, remove_columns=train_data.column_names)
formatted_val = val_data.map(format_instruction, remove_columns=val_data.column_names)

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
tokenizer.pad_token = tokenizer.eos_token

# ✅ Tokenization
def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")

tokenized_train = formatted_train.map(tokenize, batched=True, remove_columns=["text"])
tokenized_val = formatted_val.map(tokenize, batched=True, remove_columns=["text"])

# ✅ Prepare labels
def prepare_labels(example):
    example["labels"] = example["input_ids"].copy()
    return example

tokenized_train = tokenized_train.map(prepare_labels, batched=True)
tokenized_val = tokenized_val.map(prepare_labels, batched=True)

# # Add before model loading
# import torch
# torch.backends.cuda.enable_flash_sdp(False)  # Disable flash attention
# torch.backends.cuda.enable_mem_efficient_sdp(False)  # Disable memory-efficient attention

# ✅ Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# ✅ Prepare PEFT with LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj", "o_proj"]  # Safer on T4
)

model = prepare_model_for_kbit_training(base_model)
model = get_peft_model(model, peft_config)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Training args with evaluation + saving
training_args = TrainingArguments(
    output_dir="phi2-malayalam-lora",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    logging_steps=50,
    eval_steps=100,
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    gradient_checkpointing=True,
    report_to="tensorboard",
    label_names=["labels"],
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

# ✅ Start Training
trainer.train()

# ✅ Save final model + tokenizer
model.save_pretrained("phi2-malayalam-lora-final")
tokenizer.save_pretrained("phi2-malayalam-lora-final")

# Then push both:
model.push_to_hub("phi2-malayalam-lora")
tokenizer.push_to_hub("phi2-malayalam-lora")