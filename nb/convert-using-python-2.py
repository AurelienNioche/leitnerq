import os
import pickle

with open(os.path.join('data', 'mnemosyne_history.pkl'), 'rb') as f:
    history = pickle.load(f)
    history.data.to_csv("bkp.csv")
