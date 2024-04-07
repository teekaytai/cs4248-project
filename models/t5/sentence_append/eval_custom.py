from datasets import Dataset
from models.custom_utils import get_custom_tests
from models.t5.sentence_append.model import generate

custom_tests = get_custom_tests()
dataset_test = Dataset.from_list(custom_tests)

model_path = 'outputs/model_sentence_append/2_0/checkpoint-7500'
generated_sentences = generate(model_path, dataset_test, 2, 0)

with open(model_path + '/custom.txt', 'w') as f:
    for line in generated_sentences:
        f.write(f"{line}\n")
