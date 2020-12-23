# Neural Attentive Decision Ensembles
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/xl402/neural-attentive-decision-ensembles/nade)

Combine Neural Oblivious Decision Ensembles (NODE) with a compressive
attention-based memory, we have NADE.


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
