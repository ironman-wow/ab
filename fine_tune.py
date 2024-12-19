#instruction

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
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import numpy as np
from scipy.stats import pearsonr
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score
from sacrebleu.metrics import BLEU
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

# Set environment variables
os.environ["WANDB_DISABLED"] = "true"

def fine_tune_unsloth_model(dataset_dir,test_size = 0.05, seed=42):
    """
    Fine-tunes the Unsloth model using a dataset stored in the specified directory.

    Args:
        dataset_dir (str): Directory containing the dataset for fine-tuning.

    Returns:
        tuple: A tuple containing the fine-tuned model, tokenizer, and training statistics.

    Raises:
        FileNotFoundError: If the dataset directory is not found.
        ValueError: If there are issues with the model or training configuration.
        Exception: For any other exceptions that occur during fine-tuning.
    """
    try:
        # Configuration settings
        max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
        dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

        # Load the model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        )

        def formatting_prompts_func(examples):
            """
            Formats the prompts for fine-tuning by adding the question and response structure.

            Args:
                examples (dict): A dictionary containing questions and responses.

            Returns:
                dict: A dictionary containing the formatted text.
            """
            alpaca_prompt = """You will be provided with a question. Your task is to generate the correct response based on the question.

            ### Question:
            {}

            ### Response:
            {}"""
            EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
            questions = examples["question"]
            responses = examples["response"]
            texts = []
            for question, response in zip(questions, responses):
                text = alpaca_prompt.format(question, response) + EOS_TOKEN
                texts.append(text)
            return {"text": texts}

        # Load and preprocess the dataset
        dataset = load_dataset(dataset_dir)
        if isinstance(dataset, dict):
            dataset = Dataset.from_dict(dataset["train"].to_dict())

        split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
        train_dataset = split_dataset['train']
        train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
        test_dataset = split_dataset['test']
        test_dataset = test_dataset.map(formatting_prompts_func, batched=True)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        # Apply PEFT with the model
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )

        # Initialize the trainer
        trainer = SFTTrainer(
              model=model,
              tokenizer=tokenizer,
              train_dataset=train_dataset,
              dataset_text_field="text",
              max_seq_length=max_seq_length,
              dataset_num_proc=2,
              packing=False,  # Can make training 5x faster for short sequences.
              args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                num_train_epochs=5,  # Set this for 1 full training run.
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

    except FileNotFoundError as e:
        print(f"Error: Dataset directory not found. {e}")
        raise
    except ValueError as e:
        print(f"Error in training configuration: {e}")
        raise
    except Exception as e:
        print(f"An error occurred during fine-tuning: {e}")
        raise


def generate_response_inference(question, model, tokenizer):
    """
    Generates a response to a given question using the fine-tuned model.

    Args:
        question (str): The input question to generate a response for.
        model (torch.nn.Module): The fine-tuned model used for inference.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.

    Returns:
        str: The generated response.

    Raises:
        ValueError: If the model or tokenizer is not properly loaded.
        Exception: For any other errors during inference.
    """
    try:
        FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

        # Define the prompt template
        alpaca_prompt = """You will be provided with a question. Your task is to generate the correct response based on the question.

        ### Question:
        {}

        ### Response:
        {}"""

        # Prepare the input for inference
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    question,  # Question
                    "",  # Output - leave this blank for generation!
                )
            ], return_tensors="pt").to("cuda")

        # Generate the response
        outputs = model.generate(**inputs, max_new_tokens=4096, use_cache=True)
        response = tokenizer.batch_decode(outputs)

        return response

    except ValueError as e:
        print(f"Error: Invalid model or tokenizer configuration. {e}")
        raise
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        raise


def save_model_and_tokenizer(model, tokenizer, save_dir):
    """
    Saves the fine-tuned model and tokenizer to the specified directory.

    Args:
        model (torch.nn.Module): The fine-tuned model to be saved.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be saved.
        save_dir (str): The directory to save the model and tokenizer.

    Raises:
        OSError: If there is an issue with saving to the specified directory.
    """
    try:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model and tokenizer saved to {save_dir}")
    except OSError as e:
        print(f"Error: Could not save model or tokenizer. {e}")
        raise


def push_to_hub(model, tokenizer, hub_model_id, token):
    """
    Pushes the fine-tuned model and tokenizer to the Hugging Face Hub.

    Args:
        model (torch.nn.Module): The fine-tuned model to be pushed.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be pushed.
        hub_model_id (str): The identifier for the model on the Hugging Face Hub.
        token (str): The authentication token for accessing the Hugging Face Hub.

    Raises:
        Exception: For any errors during the upload process.
    """
    try:
        # Log in to Hugging Face
        login(token=token)

        # Push the model and tokenizer to the Hugging Face Hub
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        print(f"Model and tokenizer pushed to Hugging Face Hub: {hub_model_id}")
    except Exception as e:
        print(f"An error occurred while pushing to the Hugging Face Hub: {e}")
        raise

def calculate_nlp_metrics(ground_truth, model_responses):
    # Ensure inputs are lists of strings
    assert all(isinstance(s, str) for s in ground_truth)
    assert all(isinstance(s, str) for s in model_responses)
    assert len(ground_truth) == len(model_responses)

    # Tokenize sentences
    ground_truth_tokens = [word_tokenize(sentence) for sentence in ground_truth]
    model_response_tokens = [word_tokenize(sentence) for sentence in model_responses]

    # BLEU Score
    bleu = corpus_bleu([[ref] for ref in ground_truth_tokens], model_response_tokens)

    # METEOR Score
    meteor = np.mean([meteor_score([ref], hyp) for ref, hyp in zip(ground_truth_tokens, model_response_tokens)])

    # ROUGE Score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, hyp) for ref, hyp in zip(ground_truth, model_responses)]
    rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
    rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

    # BERTScore
    P, R, F1 = score(model_responses, ground_truth, lang="en", verbose=False)
    bert_score = F1.mean().item()

    # SacreBLEU Score
    sacrebleu = BLEU()
    sacrebleu_score = sacrebleu.corpus_score(model_responses, [ground_truth]).score

    # Pearson Correlation
    # We need numerical values for correlation, so we'll use the ROUGE-L scores
    rougeL_scores = [scorer.score(ref, hyp)['rougeL'].fmeasure for ref, hyp in zip(ground_truth, model_responses)]
    pearson_corr, _ = pearsonr(range(len(ground_truth)), rougeL_scores)

    #similarityScore
    model_sim = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    embeddings_1 = model_sim.encode(model_responses, normalize_embeddings=True)
    embeddings_2 = model_sim.encode(ground_truth, normalize_embeddings=True)
    similarity = embeddings_1 @ embeddings_2.T
    diagonal_values = np.diag(similarity)
    # Calculate the average of the diagonal values
    average_diagonal = np.mean(diagonal_values)

    return {
        "BLEU": bleu,
        "METEOR": meteor,
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougeL,
        "BERTScore": bert_score,
        "SacreBLEU": sacrebleu_score,
        "Similarity": average_diagonal,
        "Pearson Correlation": pearson_corr
    }

# Example usage
if __name__ == "__main__":
    dataset_dir = "ironman-wow/QA_fineTuning1"
    test_size = 0.05
    seed = 42
    try:
        # Fine-tune the model
        model, tokenizer, trainer_stats,test_dataset = fine_tune_unsloth_model(dataset_dir,test_size=0.05, seed=42)
        
        # # Inference example
        # df = pd.DataFrame(test_dataset)

        # extracted_data = df['question']
        # questions = extracted_data.tolist()

        # extracted_data = df['response']
        # ground_responses = extracted_data.tolist()

        # model_response = []

        # for question in questions:   
        #   response = generate_response_inference(question, model, tokenizer)
        #   model_response.append(response)
          
        # single_list = [item[0] for item in model_response]
        # model_responses = [re.search(r'Response:(.*?)(?=###|\Z)', item, re.DOTALL).group(1).strip() 
        #      for item in single_list 
        #      if 'Response:' in item]

        # results = calculate_nlp_metrics(ground_responses, model_responses)
        # for metric, value in results.items():
        #     print(f"{metric}: {value}")

         #Save the model and tokenizer
        save_model_and_tokenizer(model, tokenizer, "QA_fine_tuned_model")
        
    #    Push to Hugging Face Hub
        push_to_hub(model, tokenizer, "ironman-wow/QA_fineTunedModel", token="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")
    
    except Exception as e:
        print(f"An error occurred in the workflow: {e}")