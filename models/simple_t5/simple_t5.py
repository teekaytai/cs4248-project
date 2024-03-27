#https://github.com/Shivanandroy/simpleT5

import pandas as pd
from simplet5 import SimpleT5
import pickle

version = 1

model = SimpleT5()
model.from_pretrained("t5", "t5-base")

train_path = "models/train.json"
eval_path = "models/dev.json"

train_df = pd.read_json(train_path)
train_df_simple = train_df[["original", "corrected"]]
train_df_simple = train_df_simple.rename(columns={"original": "source_text", "corrected": "target_text"})
train_df_simple['source_text'] = "correct: " + train_df_simple['source_text']
eval_df = pd.read_json(eval_path)
eval_df_simple = eval_df[["original", "corrected"]]
eval_df_simple = eval_df_simple.rename(columns={"original": "source_text", "corrected": "target_text"})
eval_df_simple['source_text'] = "correct: " + eval_df_simple['source_text']

model.train(train_df = train_df_simple,
            eval_df = eval_df_simple,
            source_max_token_len = 512,
            target_max_token_len = 128,
            batch_size = 16,
            max_epochs = 1,
            use_gpu = True,
            outputdir = "outputs",
            early_stopping_patience_epochs=0)


# save the iris classification model as a pickle file
model_pkl_file = f"simple_t5_model_{version}.pkl"

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(model, file)
