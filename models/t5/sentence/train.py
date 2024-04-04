from datasets import load_dataset
from models.t5.sentence.model import preprocess_dataset, train

dataset_train = load_dataset('json', data_files='datasets/wi+locness/dataset_splits/train.json', split='train')
dataset_eval = load_dataset('json', data_files='datasets/wi+locness/dataset_splits/val.json', split='train')

preprocessed_train = dataset_train.map(
    preprocess_dataset,
    batched=True,
    fn_kwargs={"source_column_name": "original", "target_column_name": "corrected"}
)
preprocessed_eval = dataset_eval.map(
    preprocess_dataset,
    batched=True,
    fn_kwargs={"source_column_name": "original", "target_column_name": "corrected"}
)

train(preprocessed_train, preprocessed_eval, 'outputs/model_sentence_2')
