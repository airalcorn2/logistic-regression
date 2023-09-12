# Logistic Regression in scikit-learn

This repository contains the [slides](https://docs.google.com/presentation/d/1hiLwnb1o62xNFSiMjSXczi8GL0CZB8GSO50h-kzrUAw/edit?usp=sharing) and code for the "Logistic Regression in scikit-learn" presentation I gave for the Data Science Society at Auburn University.

## Cloning the repository

```bash
git clone https://github.com/airalcorn2/logistic-regression.git
```

## Setting up the virtual environment

See [here](https://michaelaalcorn.medium.com/a-python-users-response-to-python-criticism-from-a-julia-perspective-720e775fd1f2) for a short intro to virtual environments.

```bash
cd logistic-regression
python3 -m venv logistic-regression
source logistic-regression/bin/activate
```

## Installing the necessary Python packages

```bash
pip3 install -r requirements.txt
```

## Running the machine learning script

```bash
python3 lr_sklearn.py
```

## Running the statistics script

```bash
python3 lr_statsmodels.py
```
