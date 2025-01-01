def prepare_data(tokenizer, max_length=256):  
    # Load IMDB dataset
    dataset = load_dataset("imdb")

    def formatting_prompts_func(examples):
        """
        Formats the prompts for fine-tuning by adding the review and sentiment structure.
        Args:
            examples (dict): A dictionary containing review text and sentiment.
        Returns:
            dict: A dictionary containing the formatted text.
        """
        alpaca_prompt = """You will be provided with a review. Your task is to generate the correct sentiment (positive or negative) based on the review.
        ### Review:
        {}
        ### Sentiment:
        {}"""
        EOS_TOKEN = tokenizer.eos_token  # Ensure EOS_TOKEN is added
        texts = []
        for text, label in zip(examples["text"], examples["label"]):
            sentiment = 'positive' if label == 1 else 'negative'
            formatted_text = alpaca_prompt.format(text, sentiment) + EOS_TOKEN
            texts.append(formatted_text)
        return {"text": texts}

    # Tokenize the dataset using the formatting function
    tokenized_dataset = dataset.map(
        formatting_prompts_func,  # Applying the formatting function instead of tokenizing
        batched=True,
        remove_columns=dataset['train'].column_names,
        batch_size=100
    )

    # Return the tokenized dataset in the same structure as before
    return tokenized_dataset
























def prepare_data(tokenizer, max_length=256):
    def formatting_func(examples):
        alpaca_prompt = """You will be provided with a question. Your task is to generate the correct response based on the question.
        ### Question:
        {}
        ### Response:
        {}"""
        
        # Format the text with question and response
        texts = [
            alpaca_prompt.format(question, response) + tokenizer.eos_token
            for question, response in zip(examples['question'], examples['response'])
        ]
        
        # Filter out empty texts
        texts = [t for t in texts if t.strip()]
        if not texts:
            return {'input_ids': [], 'attention_mask': [], 'labels': []}
        
        # Tokenize the texts
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create labels for language modeling
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized
    
    # Load and preprocess the dataset
    dataset = load_dataset(dataset_dir)
    if isinstance(dataset, dict):
        dataset = Dataset.from_dict(dataset["train"].to_dict())
    
    # Split dataset
    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset['train']
    
    # Tokenize and format dataset
    tokenized_dataset = train_dataset.map(
        formatting_func,
        batched=True,
        remove_columns=train_dataset.column_names,
        batch_size=100
    )
    
    # Filter out empty examples
    tokenized_dataset = tokenized_dataset.filter(
        lambda example: len(example['input_ids']) > 0
    )
    
    return tokenized_dataset
