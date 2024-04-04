from datasets import load_dataset
from models.t5.sentence.model import generate, get_model
from models.eval_utils import generate_m2, analyze_error_types, analyze_params

analyze_params(get_model('outputs/model_sentence_2/checkpoint-7000'))

dataset_test = load_dataset('json', data_files='datasets/wi+locness/dataset_splits/test.json', split='train')
generated_sentences = generate('outputs/model_sentence_2/checkpoint-7000', dataset_test)

generate_m2(dataset_test['original'], generated_sentences, 'outputs/model_sentence_2/checkpoint-7000/gen.m2')

analyze_error_types('datasets/wi+locness/dataset_splits/test.m2', 'outputs/model_sentence_2/checkpoint-7000/gen.m2')
