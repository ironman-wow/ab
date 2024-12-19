#InstructionRAG

# Installs Unsloth, Xformers (Flash Attention) and all other packages!
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
# pip install transformers[torch] bitsandbytes

import os
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from huggingface_hub import login


# Set environment variables
os.environ["WANDB_DISABLED"] = "true"

# Function to fine-tune the Unsloth model
def fine_tune_unsloth_model(dataset_dir,test_size=0.05,seed =42):
    """
    Fine-tunes the Unsloth model on a specified dataset.

    Args:
        dataset_dir (str): Directory path to the dataset for fine-tuning.

    Returns:
        tuple: A tuple containing the fine-tuned model, tokenizer, and training statistics.
    
    Raises:
        ValueError: If the dataset directory is invalid or the dataset cannot be loaded.
        RuntimeError: If there are issues with the GPU or CUDA environment.
        Exception: For any other unexpected errors during the fine-tuning process.
    """
    try:
        # Configuration settings
        max_seq_length = 4096
        dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True  # Use 4bit quantization to reduce memory usage.

        # Load the model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="ironman-wow/DPO_fineTunedModel",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # Function to format prompts for fine-tuning with context
        def formatting_prompts_func(examples):
            """
            Formats the prompts for fine-tuning the model.

            Args:
                examples (dict): Dictionary containing 'context', 'question', and 'answer' fields.

            Returns:
                dict: Dictionary with formatted 'text' field for fine-tuning.
            """
            raft_prompt = """You will be provided with a question and its context. Your task is to generate the correct response based on the context.

            ### Context:
            {}

            ### Question:
            {}

            ### Response:
            {}"""
            EOS_TOKEN = tokenizer.eos_token
            contexts = examples["context"]
            questions = examples["question"]
            responses = examples["answer"]
            texts = []
            for context, question, response in zip(contexts, questions, responses):
                text = raft_prompt.format(context, question, response) + EOS_TOKEN
                texts.append(text)
            return {"text": texts}

        # Load and preprocess the dataset
        dataset = load_dataset(dataset_dir)
        if isinstance(dataset, dict):
            dataset = Dataset.from_dict(dataset["train"].to_dict())

        # Split the dataset into train and test
        train_test = dataset.train_test_split(test_size=0.05, seed=42)

        train_dataset = train_test['train']
        train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

        test_dataset = train_test['test']
        test_dataset = test_dataset.map(formatting_prompts_func, batched=True)

        # Apply PEFT with the model
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Suggested values: 8, 16, 32, etc.
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        # Initialize the trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                num_train_epochs=15,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
            ),
        )

        # Train the model
        trainer_stats = trainer.train()

        return model, tokenizer, trainer_stats, test_dataset

    except ValueError as ve:
        print(f"ValueError: {ve}")
        raise
    except RuntimeError as re:
        print(f"RuntimeError: {re}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

#####################################################


################## Saving, loading fine-tuned models

def save_model_and_tokenizer(model, tokenizer, save_dir):
    """
    Saves the fine-tuned model and tokenizer to a specified directory.

    Args:
        model (FastLanguageModel): The fine-tuned language model.
        tokenizer (Tokenizer): The tokenizer associated with the model.
        save_dir (str): The directory where the model and tokenizer should be saved.

    Raises:
        OSError: If there is an issue saving the model or tokenizer to the specified directory.
    """
    try:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    except OSError as e:
        print(f"OSError: {e}")
        raise

def push_model_to_huggingface(model, tokenizer, repo_name, token):
    """
    Pushes the fine-tuned model and tokenizer to the Hugging Face Hub.

    Args:
        model (FastLanguageModel): The fine-tuned language model.
        tokenizer (Tokenizer): The tokenizer associated with the model.
        repo_name (str): The repository name on Hugging Face Hub.
        token (str): The authentication token for Hugging Face.

    Raises:
        ValueError: If the repository name or token is invalid.
        Exception: For any other errors during the push process.
    """
    try:
        login(token=token)
        model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
    except ValueError as ve:
        print(f"ValueError: {ve}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while pushing to Hugging Face: {e}")
        raise

dataset_dir = "ironman-wow/RAG_fineTuning3"
test_size = 0.2
seed = 42
model, tokenizer, trainer_stats,test_dataset = fine_tune_unsloth_model(dataset_dir,test_size=0.05,seed = 42)


# Example usage for saving and pushing the model
save_model_and_tokenizer(model, tokenizer, "RAFT_model_20")
push_model_to_huggingface(model, tokenizer, "ironman-wow/RAFT_fineTunedModel_20", "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")  # Replace with your actual token

