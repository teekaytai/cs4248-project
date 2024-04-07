from datasets import load_dataset
from models.t5.sentence_append.model import generate, get_model
from models.eval_utils import generate_m2, analyze_error_types, analyze_params

model_path = 'outputs/model_sentence_append/2_0/checkpoint-7500'

analyze_params(get_model(model_path))

dataset_test = load_dataset('json', data_files='datasets/wi+locness/dataset_splits/test.json', split='train')
generated_sentences = generate(
    model_path, 
    dataset_test,
    1,
    0
)

generate_m2(dataset_test['original'], generated_sentences, model_path + '/gen_2.m2')

analyze_error_types('datasets/wi+locness/dataset_splits/test.m2', model_path + '/gen_2.m2')
