from datasets import load_dataset
from models.t5.sentence_append.model import preprocess_dataset, train

dataset_train = load_dataset('json', data_files='datasets/wi+locness/dataset_splits/train.json', split='train')
dataset_eval = load_dataset('json', data_files='datasets/wi+locness/dataset_splits/val.json', split='train')

dataset_kwargs = {
        "source_column_name": "original", 
        "target_column_name": "corrected",
        "para_column_name": "paragraph", 
        "corr_para_column_name": "corrected_paragraph",
        "pos_column_name": "pos", 
        "prec_range": 1, 
        "post_range": 0,
        }

preprocessed_train = dataset_train.map(
    preprocess_dataset,
    batched=True,
    fn_kwargs=dataset_kwargs
)
preprocessed_eval = dataset_eval.map(
    preprocess_dataset,
    batched=True,
    fn_kwargs=dataset_kwargs
)

train(preprocessed_train, preprocessed_eval, 'outputs/model_sentence_append/1_0')
