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

## Navigate Repository
Given our diverse choice of experiments, both in terms of models and computational resources we used, we have continued our development on individual branches. Each branch contains further information on what to expect from the code in the branch - we invite you to take a look at the same!
