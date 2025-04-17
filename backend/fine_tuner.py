from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
import datasets

def fine_tune_lora(base_model="mistralai/Mistral-7B-Instruct-v0.1", dataset_path="qa.jsonl"):
    model = AutoModelForCausalLM.from_pretrained(base_model, load_in_8bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)

    data = datasets.load_dataset("json", data_files={"train": dataset_path})
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        args=TrainingArguments(
            per_device_train_batch_size=2,
            output_dir="./fine-tuned",
            logging_steps=10,
            save_steps=100,
            num_train_epochs=1,
            fp16=True
        ),
        tokenizer=tokenizer
    )
    trainer.train()
