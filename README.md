# Getting Started
Install required packages
```
pip install -r requirements.txt
```

## Dataset
- Dataset [wi+locness](https://www.cl.cam.ac.uk/research/nl/bea2019st/) is at `datasets/wi+locness`.
- `json_to_m2.py` is modified to append the position of each sentence in its paragraph for easier preprocessing.

## Preprocessing
- Find preprocessed datasets in `datasets/wi+locness/preprocessed+para`.
- Or run `preprocess.py` to generate them.
