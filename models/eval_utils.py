import errant
import spacy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import namedtuple

NOOP_EDIT = 'A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0'
nlp = spacy.load("en_core_web_sm")
annotator = errant.load('en', nlp)

def generate_m2(input_sentences, output_sentences, output_path):
    with open(output_path, 'w') as f:
        for input, output in zip(input_sentences, output_sentences):
            edits = annotator.annotate(annotator.parse(input), annotator.parse(output))
            print('S', input, file=f)
            if not edits:
                print(NOOP_EDIT, file=f)
            for edit in edits:
                print(edit.to_m2(), file=f)
            print(file=f)  # Blank divider line



EDIT_OPS = {'M': 'Missing', 'U': 'Unnecessary', 'R': 'Replacement'}
NOOP_EDIT_TYPE = 'noop'
UNK_EDIT_TYPE = 'UNK'
EDIT_TYPES = [
    'ADJ', 'ADJ:FORM', 'ADV', 'CONJ', 'CONTR', 'DET', 'MORPH',
    'NOUN', 'NOUN:INFL', 'NOUN:NUM', 'NOUN:POSS',
    'ORTH', 'OTHER', 'PART', 'PREP', 'PRON', 'PUNCT', 'SPELL',
    'VERB', 'VERB:FORM', 'VERB:INFL', 'VERB:SVA', 'VERB:TENSE', 'WO',
]

Edit = namedtuple('Edit', ['span', 'code', 'correction'])

def load_edits(m2_file_path):
    edits = []
    with open(m2_file_path, 'r') as f:
        for group in f.read().split('\n\n'):
            if not group:
                continue
            sentence, *sent_edits = group.split('\n')
            edits.append([Edit(*e[2:].split('|||')[:3]) for e in sent_edits])
    return edits

def create_error_count_df(gold_edits, output_edits):
    rows = [*EDIT_OPS.values(), *EDIT_TYPES, NOOP_EDIT_TYPE, UNK_EDIT_TYPE]
    df = pd.DataFrame(0, index=rows, columns=['TP', 'FP', 'FN'])
    for gold_sent_edits, output_sent_edits in zip(gold_edits, output_edits):
        gold_set = set(gold_sent_edits)
        out_set = set(output_sent_edits)
        classified_edits = {
            'TP': gold_set & out_set,
            'FP': out_set - gold_set,
            'FN': gold_set - out_set
        }
        for outcome, edits in classified_edits.items():
            for edit in edits:
                if edit.code in (NOOP_EDIT_TYPE, UNK_EDIT_TYPE):
                    df.loc[edit.code, outcome] += 1
                else:
                    op, type_ = edit.code.split(':', maxsplit=1)
                    df.loc[EDIT_OPS[op], outcome] += 1
                    df.loc[type_, outcome] += 1
    df['P'] = df['TP'] / (df['TP'] + df['FP'])
    df['R'] = df['TP'] / (df['TP'] + df['FN'])
    df['F0.5'] = (1 + 0.5**2) * ((df['P'] * df['R']) / (0.5**2 * df['P'] + df['R']))
    return df

def analyze_error_types(actual_path, predicted_path):
    gold_edits = load_edits(actual_path)
    output_edits = load_edits(predicted_path)
    error_df = create_error_count_df(gold_edits, output_edits)
    print(error_df)
    sns.heatmap(error_df[['P', 'R', 'F0.5']], vmin=0.0, vmax=1.0, cmap='Reds', annot=True, yticklabels=True)
    plt.show()

def analyze_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
