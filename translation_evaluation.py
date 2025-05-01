from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from evaluate import load as eval_load
import torch
import numpy as np
from transformers import default_data_collator

# ✅ Configurations
BASE_MODEL = "microsoft/phi-2"
ADAPTER_MODEL = "farazashraf/phi2-malayalam-lora"  # LoRA fine-tuned model
DATASET_PATH = "/kaggle/input/data-en-ml/coco_dataset_bi.jsonl"
MAX_LENGTH = 512
BLEU_TOKENIZE = "flores200"

# ✅ Load tokenizer FROM YOUR ADAPTER (not base model!)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# ✅ Load base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# ✅ Resize base model's embeddings to match tokenizer
base_model.resize_token_embeddings(len(tokenizer))

# ✅ Merge LoRA adapter properly
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL, is_trainable=False)

# ✅ Load and format validation dataset
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

_, val_data = load_and_format_data()

# ✅ Tokenization
def tokenize_with_masking(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    
    context_lens = [len(tokenizer(ctx)["input_ids"]) for ctx in examples["context"]]
    
    tokenized["labels"] = [
        [-100]*ctx_len + ids[ctx_len:] 
        for ctx_len, ids in zip(context_lens, tokenized["input_ids"])
    ]
    
    return tokenized

tokenized_val = val_data.map(tokenize_with_masking, batched=True, remove_columns=["text", "context"])

# ✅ Prepare BLEU metric
bleu_metric = eval_load("sacrebleu")

# ✅ BLEU computation
# def compute_metrics(preds, labels):
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

#     # Check if any tokens are out of range
#     if np.any(labels >= len(tokenizer)):
#         print("Warning: some labels are out of the tokenizer's range!")
    
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
#     decoded_preds = [p.split("### Response:")[-1].strip() for p in decoded_preds]
#     decoded_labels = [l.strip() for l in decoded_labels]
    
#     return bleu_metric.compute(
#         predictions=decoded_preds,
#         references=[[l] for l in decoded_labels],
#         tokenize=BLEU_TOKENIZE
#     )

def compute_metrics(preds, labels):
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Set a maximum token ID value to prevent overflow
    max_token_id = len(tokenizer) - 1
    
    # Ensure all token IDs are within valid range
    safe_preds = np.clip(preds, 0, max_token_id)
    safe_labels = np.clip(labels, 0, max_token_id)
    
    try:
        decoded_preds = tokenizer.batch_decode(safe_preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(safe_labels, skip_special_tokens=True)
        
        decoded_preds = [p.split("### Response:")[-1].strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        
        return bleu_metric.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels],
            tokenize=BLEU_TOKENIZE
        )
    except Exception as e:
        print(f"Error during decoding: {e}")
        # Try decoding one by one to identify problematic samples
        decoded_preds = []
        for pred in safe_preds:
            try:
                decoded = tokenizer.decode(pred, skip_special_tokens=True)
                decoded_preds.append(decoded.split("### Response:")[-1].strip())
            except Exception as e:
                print(f"Failed to decode prediction: {e}")
                decoded_preds.append("")
        
        decoded_labels = []
        for label in safe_labels:
            try:
                decoded = tokenizer.decode(label, skip_special_tokens=True)
                decoded_labels.append(decoded.strip())
            except Exception as e:
                print(f"Failed to decode label: {e}")
                decoded_labels.append("")
        
        return bleu_metric.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels],
            tokenize=BLEU_TOKENIZE
        )

# ✅ Evaluation loop
model.eval()
all_preds = []
all_labels = []

val_loader = torch.utils.data.DataLoader(tokenized_val, batch_size=8, collate_fn=default_data_collator)

for batch in val_loader:
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False
        )
    
    all_preds.extend(outputs.cpu().numpy())
    all_labels.extend(batch["labels"].numpy())

# ✅ Final BLEU
metrics = compute_metrics(all_preds, all_labels)
print(metrics)
