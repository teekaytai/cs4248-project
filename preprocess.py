import json
import errant
import spacy

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

output_para_paths = [
    'datasets/wi+locness/para/A.dev.json',
    'datasets/wi+locness/para/A.train.json',
    'datasets/wi+locness/para/B.dev.json',
    'datasets/wi+locness/para/B.train.json',
    'datasets/wi+locness/para/C.dev.json',
    'datasets/wi+locness/para/C.train.json',
    'datasets/wi+locness/para/N.dev.json',
]

train_path = 'datasets/wi+locness/preprocessed+para/train.json'
dev_path = 'datasets/wi+locness/preprocessed+para/dev.json'
dev_m2_path = 'datasets/wi+locness/preprocessed+para/dev.m2'
train_para_path = 'datasets/wi+locness/para/train.json'
dev_para_path = 'datasets/wi+locness/para/test.json'
source_column_name = 'original'
target_column_name = 'corrected'
pos_column_name = 'pos'
paragraph_column_name = 'paragraph'

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

def output_preprocessed_sent(path, sentences, corrected_sentences, sentence_para_positions, sentence_paragrahs):
    paragraph_idx = -1
    items = []
    for idx, sentence in enumerate(sentences):
        if int(sentence_para_positions[idx]) == 0:
            paragraph_idx += 1
        item = {
            source_column_name: sentence,
            target_column_name: corrected_sentences[idx],
            paragraph_column_name: sentence_paragrahs[paragraph_idx],
            pos_column_name: int(sentence_para_positions[idx])
        }
        items.append(item)
    
    with open(path, 'w') as f:
        json.dump(items, f, indent=2)

def output_preprocessed_para(path, paras, corrected_paras):
    items = []
    for idx, para in enumerate(paras):
        items.append({
            source_column_name: para,
            target_column_name: corrected_paras[idx]
        })
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

CONCAT_PARA_TOKEN = " <CONCAT> "

nlp = spacy.load("en_core_web_sm")
annotator = errant.load('en', nlp)
NOOP_EDIT = 'A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0'

def generate_dev_m2(dev_json_path, dev_m2_path):
    items=[]

    with open(dev_json_path, 'r') as input_f:
        items = json.load(input_f)
    
    with open(dev_m2_path, 'w') as f:
        for item in items:
            edits = annotator.annotate(annotator.parse(item[source_column_name]), annotator.parse(item[target_column_name]))
            print('S', item[source_column_name], file=f)
            if not edits:
                print(NOOP_EDIT, file=f)
            for edit in edits:
                print(edit.to_m2(), file=f)
            print(file=f)  # Blank divider line

def main():
    for idx, input_path in enumerate(input_paths):
        sentences, sentence_edits, sentence_para_positions = read_m2(input_path)
        sentence_paras = get_sentence_paragraphs(sentences, sentence_para_positions)
        concatenated_sentence_paras = [CONCAT_PARA_TOKEN.join(para) for para in sentence_paras]

        corrected_sentences = make_corrected_sentences(sentences, sentence_edits)
        corrected_sentence_paragraphs = get_sentence_paragraphs(corrected_sentences, sentence_para_positions)
        concatenated_corrected_sentence_paras = [CONCAT_PARA_TOKEN.join(para) for para in corrected_sentence_paragraphs]

        output_preprocessed_sent(output_paths[idx], sentences, corrected_sentences, sentence_para_positions, sentence_paras)
        output_preprocessed_para(output_para_paths[idx], concatenated_sentence_paras, concatenated_corrected_sentence_paras)

        print("LENGTHS:")
        study_sentence_lengths(sentences)
        study_sentence_lengths(corrected_sentences)
        study_sentence_lengths(concatenated_sentence_paras)
        study_sentence_lengths(concatenated_corrected_sentence_paras)
        
    train_paths = [k for k in output_paths if 'train' in k]
    merge_json(train_paths, train_path)
    dev_paths = [k for k in output_paths if 'dev' in k]
    merge_json(dev_paths, dev_path)
    generate_dev_m2(dev_path, dev_m2_path)
    
    train_paths = [k for k in output_para_paths if 'train' in k]
    merge_json(train_paths, train_para_path)
    dev_paths = [k for k in output_para_paths if 'dev' in k]
    merge_json(dev_paths, dev_para_path)

if __name__ == "__main__":
    main()
