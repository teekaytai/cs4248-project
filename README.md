## How Can I Explore This Branch?
Please go to the `[WI_LOCNESS]_Final_Experiments.ipynb` notebook or navigate to this Colab notebook: https://colab.research.google.com/drive/1gbe9GhIShbHhhSpDyYlgvcah5SbA0rWR?usp=sharing in order to view the experiments!

## What Can I Expect?
You may expect to see our trials of BiLSTM, made following this tutorial (https://medium.com/geekculture/neural-machine-translation-using-seq2seq-model-with-attention-9faea357d70b), as well as BART model experiments. The notebook has been sectioned to help with navigation!

> Note: There is an attempt of visualisation of attention to support analysis at the very end of the notebook. While I ran out of GPU units (despite a Colab Pro subscription and a purchase of extra units), if you have any suggestions on how visualisation of attention can be explored further to bring explainability to such projects, I would be glad to connect.

## How Can I run/try out the code?
- Go to the colab note book and create a copy
- Make sure you either upload the 'datasets' folder onto your drive (as is done in our trials) or otherwise change the relevant code to enable local execution
- Play around with the transformer values and sentence flanking levels. Train the model
-  Upin execution of the model's predictions/generate behavior on the test data, note that a gen.m2 and custom.txt file with the models outputs shall be available on the specified folder (in the code) on your Colab runtime folder section. You may run the corresponding errant commands (https://github.com/chrisjbryant/errant) to get error analysis data.
