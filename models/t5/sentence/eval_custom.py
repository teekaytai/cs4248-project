from datasets import Dataset
from models.custom_utils import get_custom_tests
from models.t5.sentence.model import generate

custom_tests = get_custom_tests()
print(custom_tests)
dataset_test = Dataset.from_list(custom_tests)

model_path = 'outputs/model_sentence_2/checkpoint-7000'
generated_sentences = generate(model_path, dataset_test)

with open(model_path + '/custom.txt', 'w') as f:
    for line in generated_sentences:
        f.write(f"{line}\n")
