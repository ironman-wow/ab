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
        #response = tokenizer.batch_decode(outputs)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        response = response.split("### Response:")[-1].strip()

        return response

    except ValueError as e:
        print(f"Error: Invalid model or tokenizer configuration. {e}")
        raise
    except Exception as e:
        print(f"An error occurred during inference: {e}")
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

max_seq_length = 4096  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ironman-wow/DPO_fineTunedModel",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

dataset = load_dataset('ironman-wow/DPO_fineTuning2')
if isinstance(dataset, dict):
    dataset = Dataset.from_dict(dataset["train"].to_dict())

split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split_dataset['train']
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
test_dataset = split_dataset['test']
test_dataset = test_dataset.map(formatting_prompts_func, batched=True)


df = pd.DataFrame(test_dataset)

extracted_data = df['prompt']
questions = extracted_data.tolist()

extracted_data = df['chosen']
ground_responses = extracted_data.tolist()


model_response = []
counter = 0
for question in questions:   
  response = generate_response_inference(question, model, tokenizer)
  # print(question)
  # print(response)
  model_response.append(response)
  counter = counter+1
  print(counter)
  
# single_list = [item[0] for item in model_response]
# model_responses = [re.search(r'Response:(.*?)(?=###|\Z)', item, re.DOTALL).group(1).strip() 
#       for item in single_list 
#       if 'Response:' in item]
# print(ground_responses)
# print(model_response)
results = calculate_nlp_metrics(ground_responses, model_response)
for metric, value in results.items():
    print(f"{metric}: {value}")