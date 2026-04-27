import os
import torch
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import (
    MllamaProcessor,
    MllamaForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dotenv import load_dotenv

load_dotenv()

def fine_tune_llama_vision(
    model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
    dataset_name="data/processed/multimodal_sft_dataset", 
    output_dir="models/llama-3.2-11b-vision-ft",
):
    # 1. Load Token for Auth
    hf_token = os.getenv("HF_TOKEN")
    
    # Check for distributed training environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    # 2. BitsAndBytes Config for 4-bit Quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load Model and Processor
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        token=hf_token,
        attn_implementation="sdpa" 
    )
    model.config.use_cache = False
    processor = MllamaProcessor.from_pretrained(model_id, token=hf_token)
    
    # Ensure pad token is set
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # 4. PEFT (LoRA) Config
    peft_config = LoraConfig(
        r=int(os.getenv("LORA_RANK", "64")),
        lora_alpha=int(os.getenv("LORA_ALPHA", "16")),
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=16, 
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=10,
        logging_steps=1,
        save_steps=50,
        bf16=True, 
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=False,
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        optim="paged_adamw_32bit",
    )

    # 6. Improved Data Collator for Mllama
    try:
        vocab_limit = model.lm_head.out_features
    except AttributeError:
        if hasattr(model, "base_model"):
            vocab_limit = model.base_model.model.lm_head.out_features
        else:
            vocab_limit = model.config.text_config.vocab_size
    
    assistant_tokens = [128006, 78191, 128007]

    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        # MllamaProcessor expects images as a list of lists of PIL images 
        # (one list per text prompt)
        images = [[ex["image"]] for ex in examples]
        
        # CRITICAL: Mllama expands 1 image token into ~1600 tokens. 
        # max_length MUST be high enough to contain these (at least 2048, 4096 recommended).
        batch = processor(
            text=texts, 
            images=images, 
            return_tensors="pt", 
            padding=True,
            max_length=4096, 
            truncation=True
        )
        
        labels = batch["input_ids"].clone()
        
        # 1. Prompt Masking
        for i in range(labels.shape[0]):
            ids_list = batch["input_ids"][i].tolist()
            header_end = -1
            for idx in range(len(ids_list) - len(assistant_tokens) + 1):
                if ids_list[idx : idx + len(assistant_tokens)] == assistant_tokens:
                    header_end = idx + len(assistant_tokens)
                    if header_end < len(ids_list) and ids_list[header_end] in [128009, 271, 10]:
                        header_end += 1
                    break
            
            if header_end != -1:
                labels[i, :header_end] = -100
        
        # 2. Mask Pad tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        # 3. Mask EVERYTHING >= vocab_limit
        labels[labels >= vocab_limit] = -100
        labels[labels < 0] = -100

        batch["labels"] = labels
        return batch

    if not os.path.exists(dataset_name):
        raise FileNotFoundError(f"Dataset not found at {dataset_name}. Run formatting script first.")
        
    train_dataset = load_from_disk(dataset_name)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        args=training_args,
    )

    # 7. Start Training
    if local_rank <= 0:
        print(f"Starting fine-tuning on {dataset_name}...")
        print(f"Vocab limit: {vocab_limit}")
    
    trainer.train()
    
    # 8. Save Model
    if local_rank <= 0:
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--dataset_name", type=str, default="data/processed/multimodal_sft_dataset")
    parser.add_argument("--output_dir", type=str, default="models/llama-3.2-11b-vision-ft")
    args = parser.parse_args()
    
    fine_tune_llama_vision(
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir
    )
