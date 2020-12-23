# Neural Attentive Decision Ensembles
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/xl402/neural-attentive-decision-ensembles/nade)

Combine Neural Oblivious Decision Ensembles (NODE) with a compressive
attention-based memory, we have NADE.

### Motivation:
DL for tabular data sucks, rule of thumb: feature engineered tabular data +
gradient boosting trees >> DL on raw or feature engineered data. Recent work
such as <a href="https://arxiv.org/abs/1909.06312">NODE</a> uses DL to mimic
the behaviour of trees and is able to achieve performance comparable with
popular tree-based methods such as lightgbm.

Time-series tabular data remains a challenge, even most competitive tree-based
models rely on forward-looking training, i.e. splitting points are decided
using the entire training data, increasing generalisation error. This work focuses on combining NODE with a attention-based memory compressor, so NODE can be trained using time-series fitting methods (such as LSTM), while having large memory capacity.


### Initial Setup
Create a Python 3 virtual environment and activate:
```
virtualenv -p python3 env
source ./env/bin/activate
```
Install requirements by running:
```
pip install -r requirements.txt
```
Then export project to python path:
```
export PYTHONPATH=$PYTHONPATH:/$PATH_TO_REPO/nade
```
To test the scripts, run `pytest` in the root directory, you may wish to
install `pytest` separately
