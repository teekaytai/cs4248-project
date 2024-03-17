import json

input_paths = [
    'datasets/wi+locness/m2+para/A.dev.gold.bea19.m2',
    'datasets/wi+locness/m2+para/A.train.gold.bea19.m2',
    'datasets/wi+locness/m2+para/B.dev.gold.bea19.m2',
    'datasets/wi+locness/m2+para/B.train.gold.bea19.m2',
    'datasets/wi+locness/m2+para/C.dev.gold.bea19.m2',
    'datasets/wi+locness/m2+para/C.train.gold.bea19.m2',
    'datasets/wi+locness/m2+para/N.dev.gold.bea19.m2',
]

output_paths = [
    'datasets/wi+locness/preprocessed+para/A.dev.json',
    'datasets/wi+locness/preprocessed+para/A.train.json',
    'datasets/wi+locness/preprocessed+para/B.dev.json',
    'datasets/wi+locness/preprocessed+para/B.train.json',
    'datasets/wi+locness/preprocessed+para/C.dev.json',
    'datasets/wi+locness/preprocessed+para/C.train.json',
    'datasets/wi+locness/preprocessed+para/N.dev.json',
]

train_path = 'datasets/wi+locness/preprocessed+para/train.json'
dev_path = 'datasets/wi+locness/preprocessed+para/dev.json'

def read_m2(path):
    sentences = []
    sentence_para_positions = []
    edits = []
    sentence_edits = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('S'):
                items = line[2:].split('|||')
                sentences.append(items[0])
                sentence_para_positions.append(items[1])
            elif line.startswith('A'):
                items = line[2:].split('|||')
                start_end_indexes = items[0].split()
                start_idx = int(start_end_indexes[0])
                end_idx = int(start_end_indexes[1])
                error_type = items[1]
                replacement = items[2]
                edits.append((start_idx, end_idx, error_type, replacement))
            else:
                sentence_edits.append(edits)
                edits = []
    return sentences, sentence_edits, sentence_para_positions

def make_corrected_sentences(sentences, sentence_edits):
    corrected_sentences = []
    for idx, sentence in enumerate(sentences):
        tokens = sentence.split()
        offset = 0
        for edit in sentence_edits[idx]:
            start_idx, end_idx, error_type, replacement = edit
            if error_type == 'noop':
                continue
            replacement_tokens = replacement.split()
            tokens[offset + start_idx : offset + end_idx] = replacement_tokens
            offset = offset - (end_idx - start_idx) + len(replacement_tokens)
        corrected_sentences.append(' '.join(tokens))
    return corrected_sentences

def get_sentence_paragraphs(sentences, sentence_para_positions):
    paras = []
    para_sentences = []
    for idx, sentence in enumerate(sentences):
        if int(sentence_para_positions[idx]) == 0:
            paras.append(para_sentences)
            para_sentences = []
        para_sentences.append(sentence)
    paras.append(para_sentences)
    return paras[1:]

def output_preprocessed_data(path, sentences, corrected_sentences, sentence_para_positions, sentence_paragrahs):
    paragraph_idx = -1
    items = []
    for idx, sentence in enumerate(sentences):
        if int(sentence_para_positions[idx]) == 0:
            paragraph_idx += 1
        item = {
            'original': sentence,
            'corrected': corrected_sentences[idx],
            'paragraph': sentence_paragrahs[paragraph_idx]
        }
        items.append(item)
    
    with open(path, 'w') as f:
        json.dump(items, f, indent=2)

def study_sentence_lengths(sentences):
    max_len = 0
    min_len = 1000
    for sentence in sentences:
        l = len(sentence.split())
        max_len = max(l, max_len)
        min_len = min(l, min_len)
    print("max_len: ", max_len)
    print("min_len: ", min_len)

def merge_json(input_paths, output_path):
    result = list()
    for p in input_paths:
        with open(p, 'r') as f:
            result.extend(json.load(f))

    with open(output_path, 'w') as output_file:
        json.dump(result, output_file, indent=2)

def main():
    for idx, input_path in enumerate(input_paths):
        sentences, sentence_edits, sentence_para_positions = read_m2(input_path)
        study_sentence_lengths(sentences)
        sentence_paragrahs = get_sentence_paragraphs(sentences, sentence_para_positions)
        corrected_sentences = make_corrected_sentences(sentences, sentence_edits)
        output_preprocessed_data(output_paths[idx], sentences, corrected_sentences, sentence_para_positions, sentence_paragrahs)
    train_paths = [k for k in output_paths if 'train' in k]
    merge_json(train_paths, train_path)
    dev_paths = [k for k in output_paths if 'dev' in k]
    merge_json(dev_paths, dev_path)

if __name__ == "__main__":
    main()
