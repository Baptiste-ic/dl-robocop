# dl-robocop

## Dataset

Before running the code to train the model, make sure to download the [Paradetox](https://github.com/s-nlp/paradetox/tree/main/paradetox) dataset
(only the file `paradetox.tsv` is needed). Once downloaded, put the file in a folder named `data` at the root of the project.

## Training

To train the model from scratch, simply run `main.py` without specifying any parameters.
To train the model from an existing checkpoint, run `main.py --checkpoint_dir <path> --checkpoint_model <model weights filename> --checkpoint_optimizer <optimizer weights filename>`,
with the directory and filenames of the weights.

## Inferences

To do inferences on a trained model, run `main.py` with the same parameters as above (you don't need to specify the optimizer) and add the argument `--inference`.
Then, once the weights are loaded, you can write input sentences in your terminal and the model output will be displayed right away.
