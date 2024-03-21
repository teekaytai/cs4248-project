# Baseline model
# Train a T5 model
# Input: Error sentences
# Output: Corrected sentences

import torch
from datasets import load_dataset
from preprocess import train_path, dev_path, source_column_name, target_column_name
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer

# 1. Load dataset
dataset_train = load_dataset('json', data_files=train_path, split='train')
dataset_dev = load_dataset('json', data_files=dev_path, split='train')

# 2. Define the model
T5_SIZE = 'google-t5/t5-small'
tokenizer = T5Tokenizer.from_pretrained(T5_SIZE)
model = T5ForConditionalGeneration.from_pretrained(T5_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Tokenize

# Source refers to an error sentence. Target refers to a corrected sentence.
# Max length of train source sentence is ~220, and train target sentence is ~236.
# If test sentence has larger lengths, we may need to increase max length.
# Larger lengths are more computationally intensive.
MAX_SOURCE_LENGTH = 240
MAX_TARGET_LENGTH = 240

# T5 pretraining used this prefix to make corrections to inputs.
TASK_PREFIX = 'rectify'
TOKENIZER_PADDING = 'max_length'

def tokenize_source_sentences(sentences):
    return tokenizer(
        [TASK_PREFIX + sentence for sentence in sentences],
        padding = TOKENIZER_PADDING,
        max_length = MAX_SOURCE_LENGTH,
        truncation = True,
        return_tensors="pt",
    )

def tokenize_target_sentences(sentences):
    tokenized = tokenizer(
        sentences,
        padding = TOKENIZER_PADDING,
        max_length = MAX_TARGET_LENGTH,
        truncation = True,
        return_tensors="pt",
    )
    # Replace padding token id's of the labels by -100 so it's ignored by the loss
    ids = tokenized.input_ids
    ids[ids == tokenizer.pad_token_id] = -100
    tokenized.input_ids = ids
    return tokenized

def make_fit_input(dataset):
    tokenized_source = tokenize_source_sentences(dataset[source_column_name])
    tokenized_target = tokenize_target_sentences(dataset[target_column_name])
    input = {}
    input['input_ids'] = tokenized_source['input_ids']
    input['attention_mask'] = tokenized_source['attention_mask']
    input['labels'] = tokenized_target['input_ids']
    return input

input_train = dataset_train.map(
    make_fit_input,
    batched=True,
    num_proc=8
)
input_dev = dataset_dev.map(
    make_fit_input,
    batched=True,
    num_proc=8
)

# 4. Train model

# Define training arguments
EPOCHS = 10
OUTPUT_DIR = 'outputs/model_sentence'
BATCH_SIZE = 16
# Hugging face documentation reccomends 1e-4 or 3e-4 for T5
LEARNING_RATE = 3e-4
training_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    num_train_epochs = EPOCHS,
    evaluation_strategy = 'steps',
    eval_steps = 500,
    save_steps = 500,
    learning_rate = LEARNING_RATE,
    load_best_model_at_end = True,
    save_total_limit = 2,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = input_train,
    eval_dataset = input_dev
)

trainer.train()
