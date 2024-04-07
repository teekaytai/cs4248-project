'''
Adding pre/post k sentences to the target sentence
Input: Error sentence + k pre + j post
Output: Corrected sentence
'''

import spacy
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer

# Configs
T5_MODEL = 'google-t5/t5-small'
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512
NUM_EPOCHS = 10
# Hugging face documentation reccomends 1e-4 or 3e-4 for T5
LEARNING_RATE = 3e-4
NUM_BEAMS = 5
CONCAT_PARA_TOKEN = ' <cct> '

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)
nlp = spacy.load("en_core_web_sm")
spacy_tokenizer = nlp.tokenizer

def tokenize_source_sentences(sentences, paragraphs, sentence_positions, prec_range, post_range):
    concatenated_sentences = []
    for sentence, para, pos in zip(sentences, paragraphs, sentence_positions):
        para_len = len(para)
        concatenated = CONCAT_PARA_TOKEN.join(para[max(pos - prec_range, 0) : min(pos + post_range + 1, para_len)])
        concatenated_sentences.append(concatenated)

    return tokenizer(
        concatenated_sentences,
        padding = 'max_length',
        max_length = MAX_SOURCE_LENGTH,
        truncation = True,
        return_tensors = "pt",
    ).to(device)

def tokenize_target_sentences(sentences, paragraphs, sentence_positions, prec_range, post_range):
    concatenated_sentences = []
    for sentence, para, pos in zip(sentences, paragraphs, sentence_positions):
        para_len = len(para)
        concatenated = CONCAT_PARA_TOKEN.join(para[max(pos - prec_range, 0) : min(pos + post_range + 1, para_len)])
        concatenated_sentences.append(concatenated)

    tokenized = tokenizer(
        concatenated_sentences,
        padding = 'max_length',
        max_length = MAX_TARGET_LENGTH,
        truncation = True,
        return_tensors="pt",
    ).to(device)
    # Replace padding token ids of the labels by -100 so it's ignored by the loss
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


def preprocess_dataset(dataset, source_column_name, target_column_name, para_column_name, corr_para_column_name, pos_column_name, prec_range, post_range):
    tokenized_source = tokenize_source_sentences(dataset[source_column_name], dataset[para_column_name], dataset[pos_column_name], prec_range, post_range)
    tokenized_target = tokenize_target_sentences(dataset[target_column_name], dataset[corr_para_column_name], dataset[pos_column_name], prec_range, post_range)
    input = {}
    input['input_ids'] = tokenized_source['input_ids']
    input['attention_mask'] = tokenized_source['attention_mask']
    input['labels'] = tokenized_target['input_ids']
    return input

def train(train_dataset, eval_dataset, output_dir):
    model = T5ForConditionalGeneration.from_pretrained(T5_MODEL)
    model.to(device)

    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
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

def generate(model_path, dataset, prec_range, post_range):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)

    generated_sentences = []

    for sample in dataset:
        original = sample['original']
        para = sample['paragraph']
        pos = sample['pos']
        tokenized = tokenize_source_sentences([original], [para], [pos], prec_range, post_range)
        generated = model.generate(
            tokenized.input_ids,
            max_length = MAX_TARGET_LENGTH,
            num_beams = NUM_BEAMS,
            early_stopping=True
        )
        generated_chunk = tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        )
        print(generated_chunk)
        generated_parts = generated_chunk.split("cct>")
        range_len = prec_range + post_range + 1
        if len(generated_parts) == 1:
            generated_sentence = ' '.join(tok.text for tok in spacy_tokenizer(generated_parts[0])).strip()
        elif len(generated_parts) == range_len:
            generated_sentence = ' '.join(tok.text for tok in spacy_tokenizer(generated_parts[prec_range])).strip()
        elif pos - prec_range < 0:
            generated_sentence = ' '.join(tok.text for tok in spacy_tokenizer(generated_parts[pos])).strip()
        elif pos + post_range >= len(para):
            pos_from_back = pos + post_range + 1 - len(para)
            generated_sentence = ' '.join(tok.text for tok in spacy_tokenizer(generated_parts[-pos_from_back])).strip()
        else:
            generated_sentence = ' '.join(tok.text for tok in spacy_tokenizer(generated_parts[-1])).strip()

        print("original: ", original)
        print("generated: ", generated_sentence)
        # Retokenize sentence using spacy to restore correct spacing between tokens
        # for accurate error correction score calculation
        generated_sentences.append(generated_sentence)
    
    return generated_sentences

def get_model(path):
    model = T5ForConditionalGeneration.from_pretrained(path)
    model.to(device)

    return model
