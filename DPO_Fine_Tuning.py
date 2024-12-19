#DPOTuning

# Installs Unsloth, Xformers (Flash Attention) and all other packages!
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
# pip install transformers[torch] bitsandbytes


import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from huggingface_hub import login
from datasets import Dataset
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from trl import DPOTrainer, DPOConfig
from unsloth import PatchDPOTrainer
PatchDPOTrainer()

# Set environment variables
os.environ["WANDB_DISABLED"] = "true"

def fine_tune_unsloth_model(dataset_dir,test_size=0.05,seed = 42):
    """
    Fine-tunes the Unsloth language model using the provided dataset.

    Args:
        dataset_dir (str): The directory containing the dataset for fine-tuning.

    Returns:
        tuple: A tuple containing the fine-tuned model, tokenizer, and training statistics.
    """
    try:
        # Configuration settings
        max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "ironman-wow/QA_fineTunedModel", # Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )

        def formatting_prompts_func(examples):
            """
            Formats prompts for fine-tuning by applying a DPO prompt template.

            Args:
                examples (dict): A dictionary containing prompt examples with keys 'prompt', 'chosen', and 'rejected'.

            Returns:
                dict: A dictionary with formatted text data.
            """
            dpo_prompt_template = """You will be provided with a question and two responses. Your task is to generate a response that aligns with the qualities of the better response, while avoiding the shortcomings of the weaker response.

            ### Question:
            {}

            ### Better Response (Preferred):
            {}

            ### Weaker Response (Not Preferred):
            {}
            """
            EOS_TOKEN = tokenizer.eos_token  # Ensure you add EOS_TOKEN to mark the end of the text
            prompts = examples["prompt"]
            chosens = examples["chosen"]
            rejecteds = examples["rejected"]
            texts = []
            for prompt, chosen, rejected in zip(prompts, chosens, rejecteds):
                text = dpo_prompt_template.format(prompt, chosen, rejected) + EOS_TOKEN
                texts.append(text)
            return {"text": texts}

        # Load and preprocess the dataset
        dataset = load_dataset(dataset_dir)
        if isinstance(dataset, dict):
            dataset = Dataset.from_dict(dataset["train"].to_dict())

        # Split the dataset into train and test
        train_test = dataset.train_test_split(test_size=0.05, seed =42)

        train_dataset = train_test['train']
        train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

        test_dataset = train_test['test']
        test_dataset = test_dataset.map(formatting_prompts_func, batched=True)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Currently only supports dropout = 0
            bias = "none",    # Currently only supports bias = "none"
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
        
        # Initialize the trainer
        dpo_trainer = DPOTrainer(
            model = model,
            ref_model = None,
            args = DPOConfig(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_ratio = 0.1,
                num_train_epochs = 1,
                learning_rate = 2e-4,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
            ),
            beta = 0.1,
            train_dataset = train_dataset,
            # eval_dataset = raw_datasets["test"],
            tokenizer = tokenizer,
            max_length = 2048,
            
        )
      
        # Train the model
        trainer_stats = dpo_trainer.train()

        # Return the model, tokenizer, and training statistics
        return model, tokenizer, trainer_stats,test_dataset

    except FileNotFoundError as e:
        print(f"Error: The dataset directory was not found. {e}")
    except torch.cuda.CudaError as e:
        print(f"Error: CUDA-related issue encountered. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def save_model_and_tokenizer(model, tokenizer, directory):
    """
    Saves the fine-tuned model and tokenizer to the specified directory.

    Args:
        model (PreTrainedModel): The fine-tuned model to be saved.
        tokenizer (PreTrainedTokenizer): The tokenizer to be saved.
        directory (str): The directory to save the model and tokenizer.
    """
    try:
        model.save_pretrained(directory)  # Local saving
        tokenizer.save_pretrained(directory)
    except Exception as e:
        print(f"Error occurred while saving the model and tokenizer: {e}")

def push_model_to_hub(model, tokenizer, model_name):
    """
    Pushes the model and tokenizer to the Hugging Face Hub.

    Args:
        model (PreTrainedModel): The fine-tuned model to be pushed.
        tokenizer (PreTrainedTokenizer): The tokenizer to be pushed.
        model_name (str): The name under which the model will be pushed to the Hub.
    """
    try:
        login(token="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")  # Replace with your token
        model.push_to_hub(model_name)
        tokenizer.push_to_hub(model_name)
    except Exception as e:
        print(f"Error occurred while pushing the model to the Hub: {e}")


   
# Example usage
dataset_dir = "ironman-wow/DPO_fineTuning2"
test_size = 0.05
seed =42
model, tokenizer, trainer_stats,test_dataset = fine_tune_unsloth_model(dataset_dir,test_size=0.05, seed =42)

# Save model and tokenizer locally
save_model_and_tokenizer(model, tokenizer, "dpo_model")

# Push the model and tokenizer to Hugging Face Hub
push_model_to_hub(model, tokenizer, "ironman-wow/test_model")


