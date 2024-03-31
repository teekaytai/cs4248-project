'''Sentence with preceeding sentences
Train a T5 model
Input: Error sentences with their preceeding sentences concatenated
Output: Corrected sentences

Things to tune:
1. Number of preceeding sentences to consider in the paragraph context
(The rest are not so important)
2. MAX_SOURCE_LENGTH
3. MAX_TARGET_LENGTH'
4. Training arguments
  a. Mainly: Number of epochs, batch size
  b. Less important: Learning rate
5. Try other transformer models
'''

import torch
from datasets import load_dataset
from preprocess import train_path, dev_path, source_column_name, target_column_name, paragraph_column_name, pos_column_name, CONCAT_PARA_TOKEN
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
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 240

TASK_PREFIX = 'rectify: '
TOKENIZER_PADDING = 'max_length'
PRECEEDING_RANGE = 2

def tokenize_source_sentences(sentences, paragraphs, sentence_positions):
    concatenated_sentences = []
    for idx, sentence in enumerate(sentences):
        sent_pos = sentence_positions[idx]
        para = paragraphs[idx]
        concatenated_sentences.append(CONCAT_PARA_TOKEN.join(para[max(sent_pos - PRECEEDING_RANGE, 0) : sent_pos + 1]))
    return tokenizer(
        [TASK_PREFIX + sentence for sentence in concatenated_sentences],
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
    tokenized_source = tokenize_source_sentences(dataset[source_column_name], dataset[paragraph_column_name], dataset[pos_column_name])
    tokenized_target = tokenize_target_sentences(dataset[target_column_name])
    input = {}
    input['input_ids'] = tokenized_source['input_ids']
    input['attention_mask'] = tokenized_source['attention_mask']
    input['labels'] = tokenized_target['input_ids']
    return input

input_train = dataset_train.map(
    make_fit_input,
    batched=True,
)
input_dev = dataset_dev.map(
    make_fit_input,
    batched=True,
)

# 4. Train model

# Define training arguments
EPOCHS = 10
OUTPUT_DIR = 'outputs/model_sentence_prec'
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
