'''
Baseline
Input: Error Sentences
Output: Corrected Sentences
'''
import spacy
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer

# Configs
T5_MODEL = 'google-t5/t5-small'
TASK_PREFIX = 'rectify: '
MAX_SOURCE_LENGTH = 240
MAX_TARGET_LENGTH = 240
NUM_EPOCHS = 10
# Hugging face documentation reccomends 1e-4 or 3e-4 for T5
LEARNING_RATE = 3e-4
NUM_BEAMS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)
nlp = spacy.load("en_core_web_sm")
spacy_tokenizer = nlp.tokenizer

def tokenize_source_sentences(sentences):
    return tokenizer(
        [TASK_PREFIX + sentence for sentence in sentences],
        padding = 'max_length',
        max_length = MAX_SOURCE_LENGTH,
        truncation = True,
        return_tensors = "pt",
    ).to(device)

def tokenize_target_sentences(sentences):
    tokenized = tokenizer(
        sentences,
        padding = 'max_length',
        max_length = MAX_TARGET_LENGTH,
        truncation = True,
        return_tensors="pt",
    ).to(device)
    # Replace padding token id's of the labels by -100 so it's ignored by the loss
    ids = tokenized.input_ids
    ids[ids == tokenizer.pad_token_id] = -100
    tokenized.input_ids = ids
    return tokenized

def preprocess_dataset(dataset, source_column_name, target_column_name):
    tokenized_source = tokenize_source_sentences(dataset[source_column_name])
    tokenized_target = tokenize_target_sentences(dataset[target_column_name])
    input = {}
    input['input_ids'] = tokenized_source['input_ids']
    input['attention_mask'] = tokenized_source['attention_mask']
    input['labels'] = tokenized_target['input_ids']
    return input

def train(train_dataset, eval_dataset, output_dir):
    model = T5ForConditionalGeneration.from_pretrained(T5_MODEL)
    model.to(device)

    training_args = TrainingArguments(
        output_dir = output_dir,
        num_train_epochs = NUM_EPOCHS,
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
        train_dataset = train_dataset,
        eval_dataset = eval_dataset
    )

    trainer.train()

def generate(model_path, dataset):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)

    generated_sentences = []

    for sample in dataset:
        original = sample['original']
        tokenized = tokenize_source_sentences([original])
        generated = model.generate(
            tokenized.input_ids,
            max_length = MAX_TARGET_LENGTH,
            num_beams = NUM_BEAMS,
            early_stopping=True
        )
        generated_sentence = tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        )
        # Retokenize sentence using spacy to restore correct spacing between tokens
        # for accurate error correction score calculation
        generated_sentence = ' '.join(tok.text for tok in spacy_tokenizer(generated_sentence))
        generated_sentences.append(generated_sentence)
    
    return generated_sentences

def get_model(path):
    model = T5ForConditionalGeneration.from_pretrained(path)
    model.to(device)

    return model
