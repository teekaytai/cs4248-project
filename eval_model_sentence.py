import torch
from datasets import load_dataset
from preprocess import dev_path
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
import spacy
import errant

dataset_test = load_dataset('json', data_files=dev_path, split='train')

T5_MODEL = 'google-t5/t5-small'
MODEL_PATH = 'outputs/model_sentence/checkpoint-8000'
OUT_M2_PATH = MODEL_PATH + '/generated.m2'
tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

nlp = spacy.load("en_core_web_sm")
spacy_tokenizer = nlp.tokenizer
annotator = errant.load('en', nlp)

MAX_SOURCE_LENGTH = 240
MAX_TARGET_LENGTH = 240
TASK_PREFIX = 'rectify: '
GEN_NUM_BEAMS = 5

def generate_correction(model, tokenizer, sample):
    input_text = f"{TASK_PREFIX}{sample['original']}"
    inputs = tokenizer.encode(
        input_text,
        max_length=MAX_SOURCE_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    ).to(device)
    corrected_ids = model.generate(
        inputs,
        max_length=MAX_TARGET_LENGTH,
        num_beams=GEN_NUM_BEAMS,
        early_stopping=True,
    )
    corrected_sentence = tokenizer.decode(
        corrected_ids[0],
        skip_special_tokens=True,
    )
    # Retokenize sentence using spacy to restore correct spacing between tokens
    # for accurate error correction score calculation
    corrected_sentence = ' '.join(tok.text for tok in spacy_tokenizer(corrected_sentence))
    return corrected_sentence

NOOP_EDIT = 'A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0'

# Can use later for analysing performance for each type of error
output_edit_types = []

with open(OUT_M2_PATH, 'w') as f:
    for sample in dataset_test:
        orig = sample['original']
        corrected = generate_correction(model, tokenizer, sample)
        edits = annotator.annotate(annotator.parse(orig), annotator.parse(corrected))
        output_edit_types.append([edit.type for edit in edits])
        print('S', orig, file=f)
        if not edits:
            print(NOOP_EDIT, file=f)
        for edit in edits:
            print(edit.to_m2(), file=f)
        print(file=f)  # Blank divider line
