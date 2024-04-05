import json

from sklearn.model_selection import train_test_split

train_input_paths = [
    'datasets/wi+locness/m2+para/A.train.gold.bea19.m2',
    'datasets/wi+locness/m2+para/B.train.gold.bea19.m2',
    'datasets/wi+locness/m2+para/C.train.gold.bea19.m2',
]

dev_input_paths = [
    'datasets/wi+locness/m2+para/A.dev.gold.bea19.m2',
    'datasets/wi+locness/m2+para/B.dev.gold.bea19.m2',
    'datasets/wi+locness/m2+para/C.dev.gold.bea19.m2',
    'datasets/wi+locness/m2+para/N.dev.gold.bea19.m2',
]

train_json_path = 'datasets/wi+locness/dataset_splits/train.json'
val_json_path = 'datasets/wi+locness/dataset_splits/val.json'
test_json_path = 'datasets/wi+locness/dataset_splits/test.json'
train_m2_path = 'datasets/wi+locness/dataset_splits/train.m2'
val_m2_path = 'datasets/wi+locness/dataset_splits/val.m2'
test_m2_path = 'datasets/wi+locness/dataset_splits/test.m2'

test_size = 0.2

def read_m2(path):
    # Dataset grouped by paragraphs so that sentences from the same paragraph
    # do not get split between training and validation sets
    dataset = []
    sentences = []
    sentence_edits = []
    para_edits = []
    sentence_m2_lines = []
    para_m2_lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('S'):
                items = line[2:].split('|||')
                if int(items[1]) == 0:
                    # Start new paragraph
                    sentences = []
                    para_edits = []
                    para_m2_lines = []
                    dataset.append((sentences, para_edits, para_m2_lines))
                sentences.append(items[0])
                sentence_edits = []
                para_edits.append(sentence_edits)
                sentence_m2_lines = ['S ' + items[0]]
                para_m2_lines.append(sentence_m2_lines)
            elif line.startswith('A'):
                items = line[2:].split('|||')
                start_end_indexes = items[0].split()
                start_idx = int(start_end_indexes[0])
                end_idx = int(start_end_indexes[1])
                error_type = items[1]
                replacement = items[2]
                sentence_edits.append((start_idx, end_idx, error_type, replacement))
                sentence_m2_lines.append(line.strip())
    return dataset

def make_corrected_sentence(sentence, sentence_edits):
    tokens = sentence.split()
    offset = 0
    for edit in sentence_edits:
        start_idx, end_idx, error_type, replacement = edit
        if error_type == 'noop':
            continue
        replacement_tokens = replacement.split()
        tokens[offset + start_idx : offset + end_idx] = replacement_tokens
        offset = offset - (end_idx - start_idx) + len(replacement_tokens)
    return ' '.join(tokens)

def output_preprocessed_data(json_path, m2_path, dataset):
    items = []
    for paragraph, para_edits, _ in dataset:
        corrected_paragraph = []
        for i, (sentence, sentence_edits) in enumerate(zip(paragraph, para_edits)):
            corrected_paragraph.append(make_corrected_sentence(sentence, sentence_edits))
        for i, (sentence, sentence_edits) in enumerate(zip(paragraph, para_edits)):
            item = {
                'original': sentence,
                'corrected': corrected_paragraph[i],
                'paragraph': paragraph,
                'corrected_paragraph': corrected_paragraph,
                'pos': i,
            }
            items.append(item)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2)

    with open(m2_path, 'w', encoding='utf-8') as f:
        for _, _, para_m2_lines in dataset:
            for sentence_m2_lines in para_m2_lines:
                print('\n'.join(sentence_m2_lines), file=f, end='\n\n')

def main():
    train_dataset = []
    types = []
    for i, input_path in enumerate(train_input_paths):
        dataset = read_m2(input_path)
        train_dataset.extend(dataset)
        types.extend([i] * len(dataset))

    train_dataset, val_dataset = train_test_split(train_dataset, test_size=test_size, random_state=4248, stratify=types)
    output_preprocessed_data(train_json_path, train_m2_path, train_dataset)
    output_preprocessed_data(val_json_path, val_m2_path, val_dataset)

    test_dataset = []
    for input_path in dev_input_paths:
        dataset = read_m2(input_path)
        test_dataset.extend(dataset)
    output_preprocessed_data(test_json_path, test_m2_path, test_dataset)

if __name__ == "__main__":
    main()
