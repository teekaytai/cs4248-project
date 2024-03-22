import torch
import torch.nn as nn
from datasets import load_dataset
from preprocess import train_path, dev_path, source_column_name, target_column_name, paragraph_column_name
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer 

class DoubleEncNet(nn.Module):
    def __init__(self, sent_enc_model, para_enc_model, dec_model, para_enc_hidden_size, dec_hidden_size):
        super(DoubleEncNet, self).__init__()
        self.sent_enc_model = sent_enc_model
        self.para_enc_model = para_enc_model
        self.dec_model = dec_model
        hidden_size = 768
        self.dim_adjustor = nn.Sequential(
            nn.Linear(para_enc_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dec_hidden_size)
        )

    def forward(self, sent_input_ids, sent_attention_mask, para_input_ids, para_attention_mask, labels):
        sent_enc = self.sent_enc_model.encoder(
            input_ids=sent_input_ids,
            attention_mask=sent_attention_mask,
            return_dict=True
        ).last_hidden_state
        para_enc = self.para_enc_model(
            input_ids=para_input_ids,
            attention_mask=para_attention_mask,
            return_dict=True
        ).last_hidden_state
        # TODO: Determine how to best combine sent_enc and para_enc
        # Currently this concatenates para hidden states to sentence hidden states
        para_enc = self.dim_adjustor(para_enc)
        enc = torch.cat((sent_enc, para_enc), dim = 1)
        output = self.dec_model.decoder(labels, encoder_hidden_states=enc).last_hidden_state
        output = output * (self.dec_model.model_dim ** -0.5)
        logits = self.dec_model.lm_head(output)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index = -100)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).unsqueeze(dim = 0)
        return loss

# 1. Load dataset
dataset_train = load_dataset('json', data_files=train_path, split='train')
dataset_dev = load_dataset('json', data_files=dev_path, split='train')

# 2. Define the model
T5_SIZE = 'google-t5/t5-small'
t5_tokenizer = T5Tokenizer.from_pretrained(T5_SIZE)
sent_enc_model = T5ForConditionalGeneration.from_pretrained(T5_SIZE)
dec_model = T5ForConditionalGeneration.from_pretrained(T5_SIZE)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
para_enc_model = BertModel.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sent_enc_model.to(device)
para_enc_model.to(device)
dec_model.to(device)

para_enc_hidden_size = 768 # bert encoder output hidden size
dec_hidden_size = 512 # t5 encoder output hidden size

model = DoubleEncNet(sent_enc_model, para_enc_model, dec_model, para_enc_hidden_size, dec_hidden_size)

# 3. Tokenize sentences and paragraphs
MAX_SOURCE_SENT_LENGTH = 240
MAX_TARGET_SENT_LENGTH = 240
MAX_SOURCE_PARA_LENGTH = 512

TASK_PREFIX = 'rectify'
TOKENIZER_PADDING = 'max_length'

def tokenize_source_sentences(sentences):
    return t5_tokenizer(
        [TASK_PREFIX + sentence for sentence in sentences],
        padding = TOKENIZER_PADDING,
        max_length = MAX_SOURCE_SENT_LENGTH,
        truncation = True,
        return_tensors="pt",
    )

def tokenize_target_sentences(sentences):
    tokenized = t5_tokenizer(
        sentences,
        padding = TOKENIZER_PADDING,
        max_length = MAX_TARGET_SENT_LENGTH,
        truncation = True,
        return_tensors="pt",
    )
    # Replace padding token id's of the labels by -100 so it's ignored by the loss
    ids = tokenized.input_ids
    ids[ids == t5_tokenizer.pad_token_id] = -100
    tokenized.input_ids = ids
    return tokenized

def tokenize_source_paragraphs(paragraphs):
    return bert_tokenizer(
        paragraphs,
        padding = TOKENIZER_PADDING,
        max_length = MAX_SOURCE_PARA_LENGTH,
        truncation = True,
        return_tensors="pt"
    )

def make_fit_input(dataset):
    tokenized_source_sent = tokenize_source_sentences(dataset[source_column_name])
    tokenized_target_sent = tokenize_target_sentences(dataset[target_column_name])
    tokenized_source_para = tokenize_source_paragraphs(dataset[paragraph_column_name])
    input = {}
    input['sent_input_ids'] = tokenized_source_sent['input_ids']
    input['sent_attention_mask'] = tokenized_source_sent['attention_mask']
    input['para_input_ids'] = tokenized_source_para['input_ids']
    input['para_attention_mask'] = tokenized_source_para['attention_mask']
    input['labels'] = tokenized_target_sent['input_ids']
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
OUTPUT_DIR = 'outputs/model_double_enc'
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
