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
from evaluate import load as eval_load
import numpy as np

# Add this at the VERY TOP to prevent dynamo issues
# import os
# os.environ["TORCHDYNAMO_DISABLE"] = "1"

# ✅ Configuration
MODEL_NAME = "microsoft/phi-2"
DATASET_PATH = "/kaggle/input/data-en-ml/coco_dataset_bi.jsonl"
MAX_LENGTH = 512
BLEU_TOKENIZE = "flores200"  # For Malayalam compatibility

# ✅ Load and prepare dataset
def load_and_format_data():
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.select(range(1200)).train_test_split(test_size=0.1)
    
    def format_example(ex):
        context = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n"
        return {
            "text": context + ex["output"],
            "context": context
        }
    
    return (
        dataset["train"].map(format_example, remove_columns=dataset["train"].column_names),
        dataset["test"].map(format_example, remove_columns=dataset["test"].column_names)
    )

train_data, val_data = load_and_format_data()

# ✅ Initialize tokenizer with Malayalam support
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Add Malayalam Unicode characters to tokenizer
malayalam_tokens = [chr(i) for i in range(0x0D00, 0x0D7F)]
tokenizer.add_tokens(malayalam_tokens)

# ✅ Tokenization with label masking
def tokenize_with_masking(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    
    # Calculate context length for each example
    context_lens = [len(tokenizer(ctx)["input_ids"]) for ctx in examples["context"]]
    
    # Mask labels (context part gets -100)
    tokenized["labels"] = [
        [-100]*ctx_len + ids[ctx_len:] 
        for ctx_len, ids in zip(context_lens, tokenized["input_ids"])
    ]
    
    return tokenized

tokenized_train = train_data.map(tokenize_with_masking, batched=True, remove_columns=["text", "context"])
tokenized_val = val_data.map(tokenize_with_masking, batched=True, remove_columns=["text", "context"])

# ✅ Model setup with Phi-2 specific config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
    use_cache=False,
    torch_dtype=torch.float16  # Add this line
)
base_model.resize_token_embeddings(len(tokenizer))  # For added Malayalam tokens

# ✅ Phi-2 optimized LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["Wqkv", "out_proj", "fc1", "fc2"],  # Phi-2 specific
    modules_to_save=["lm_head"]  # Crucial for new tokens
)

model = prepare_model_for_kbit_training(base_model)
model = get_peft_model(model, peft_config)

# ✅ Evaluation metrics
bleu_metric = eval_load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Strip context from predictions
    decoded_preds = [p.split("### Response:")[-1].strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]  # Add this line
    
    return bleu_metric.compute(
        predictions=decoded_preds,
        references=[[l] for l in decoded_labels],
        tokenize=BLEU_TOKENIZE
    )

# ✅ Optimized training arguments
training_args = TrainingArguments(
    output_dir="phi2-malayalam-lora",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    optim="paged_adamw_8bit",
    warmup_steps=100,
    logging_steps=50,
    eval_steps=100,
    eval_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    gradient_checkpointing=True,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    group_by_length=True,
    label_names=["labels"]
)

# ✅ Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# ✅ Start training
trainer.train()

# ✅ Save and push
model.save_pretrained("phi2-malayalam-lora-final")
tokenizer.save_pretrained("phi2-malayalam-lora-final")

model.push_to_hub("phi2-malayalam-lora")
tokenizer.push_to_hub("phi2-malayalam-lora")