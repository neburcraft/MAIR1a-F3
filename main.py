import classifiers
import evaluate

import argparse

import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", help="enable verbose output")
    args = parser.parse_args()