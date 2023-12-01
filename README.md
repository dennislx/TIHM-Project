>📋  A template README.md for code accompanying a Machine Learning paper

# What should be my title?

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements: (TODO)

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training and Evaluation

To train the model(s) in the paper, run this command:

```train
python src/main.py -c src/experiments/24h-raw.yaml
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.


## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Dec 1st: Experiment Configuration (results/baseline/config.yaml)

|                   | loss          | accuracy      | precision     | recall        | f1score       | iou           |
|:------------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| ('GBM', 'test')   | 0.394 (0.097) | 0.840 (0.049) | 0.806 (0.097) | 0.606 (0.026) | 0.624 (0.044) | 0.520 (0.041) |
| ('GBM', 'train')  | 0.241 (0.059) | 0.917 (0.027) | 0.933 (0.030) | 0.795 (0.043) | 0.842 (0.040) | 0.744 (0.052) |
| ('GBM', 'valid')  | 0.367 (0.062) | 0.827 (0.044) | 0.697 (0.095) | 0.619 (0.019) | 0.631 (0.016) | 0.520 (0.020) |
| ('LSTM', 'test')  | 0.604 (0.047) | 0.732 (0.021) | 0.586 (0.042) | 0.661 (0.042) | 0.583 (0.053) | 0.458 (0.043) |
| ('LSTM', 'train') | 0.564 (0.016) | 0.749 (0.025) | 0.608 (0.015) | 0.702 (0.018) | 0.615 (0.022) | 0.484 (0.023) |
| ('LSTM', 'valid') | 0.601 (0.033) | 0.665 (0.066) | 0.568 (0.030) | 0.665 (0.044) | 0.534 (0.053) | 0.406 (0.047) |
| ('MLP', 'test')   | 0.558 (0.220) | 0.775 (0.074) | 0.595 (0.100) | 0.606 (0.092) | 0.600 (0.096) | 0.487 (0.070) |
| ('MLP', 'train')  | 0.280 (0.100) | 0.895 (0.050) | 0.822 (0.085) | 0.823 (0.064) | 0.817 (0.073) | 0.714 (0.099) |
| ('MLP', 'valid')  | 0.668 (0.213) | 0.700 (0.084) | 0.659 (0.030) | 0.692 (0.050) | 0.639 (0.048) | 0.489 (0.059) |


>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. (TODO)