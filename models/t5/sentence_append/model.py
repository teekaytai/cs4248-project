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
        [sentence for sentence in concatenated_sentences],
        padding = 'max_length',
        max_length = MAX_SOURCE_LENGTH,
        truncation = True,
        return_tensors = "pt",
    ).to(device)


