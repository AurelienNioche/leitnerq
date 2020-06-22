import pickle
import os

import numpy as np
import pandas as pd

from lentil import evaluate
from lentil.datatools import InteractionHistory

import mem

from tqdm import tqdm

from matplotlib import pyplot as plt

import matplotlib as mpl

mpl.rc('savefig', dpi=300)
mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsfonts}')

# with open(os.path.join('data', 'mnemosyne_history.pkl'), 'rb') as f:
#     history = pickle.load(f, encoding='latin1')
_history_data = pd.read_csv("mnemosyne-history.csv")
print("Converting the data into an InteractionHistory object...")

history = InteractionHistory(data=_history_data)

# %%

plt.xlabel(r'Delay ($\log_{10}$-seconds)')
plt.ylabel('Frequency (Number of Interactions)')
plt.hist(
    np.log10(1 + (history.data['timestamp'] - history.data['tlast']).values),
    bins=20)
plt.savefig(os.path.join('figures', 'mnemosyne_mindelay', 'delay-hist.pdf'),
            bbox_inches='tight')
plt.show()

# %%

min_delays = np.exp(np.arange(0, 16, 16 / 10))
len(min_delays)

# %%

history.data.sort_values('timestamp', inplace=True)

# %%

deck_of_student_item = [{} for _ in min_delays]

deck = [[] for _ in min_delays]
for _, ixn in tqdm(history.data.iterrows(), total=len(history.data)):
    student_item = (ixn['user_id'], ixn['module_id'])
    for i, min_delay in enumerate(min_delays):
        d = deck_of_student_item[i].get(student_item, 1)
        deck[i].append(d)

        if ixn['outcome']:
            if ixn['timestamp'] - ixn['tlast'] >= min_delay:
                deck_of_student_item[i][student_item] = d + 1
            else:
                deck_of_student_item[i][student_item] = d
        else:
            deck_of_student_item[i][student_item] = max(1, d - 1)

for i, x in enumerate(min_delays):
    history.data['deck_%d' % x] = deck[i]

# %%

# with open(os.path.join('data', 'mnemosyne_history_vMINDELAY.pkl'), 'wb') as f:
#     pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

# %% md

# Setup
# the
# IRT
# benchmark
# models and memory
# models


# %%

def meta_build_efc_model(
        strength_model='deck', using_delay=True,
        using_global_difficulty=True, debug_mode_on=True):
    def build_efc_model(history, filtered_history, split_history=None):
        model = mem.EFCModel(
            filtered_history, strength_model=strength_model,
            using_delay=using_delay,
            using_global_difficulty=using_global_difficulty,
            debug_mode_on=debug_mode_on)
        model.fit(
            # learning_rate=0.1,
            learning_rate=(10000 if not using_global_difficulty else 0.1),
            ftol=1e-6, max_iter=200)
        return model

    return build_efc_model


# %%

model_builders = {}
model_builders.update({('GMIND%d' % x): meta_build_efc_model(
    strength_model=('deck_%d' % x), using_global_difficulty=True,
debug_mode_on=False) for x in
    min_delays})
model_builders.update({('IMIND%d' % x): meta_build_efc_model(
    strength_model=('deck_%d' % x), using_global_difficulty=False,
debug_mode_on=False) for x in
    min_delays})

# %%

print(
"Number of models = %d" % (len(model_builders)))
print(
'\n'.join(model_builders.keys()))

# %% md

# Perform
# the
# evaluations

# %%
print("Computing results")
results = evaluate.cross_validated_auc(
    model_builders,
    history,
    num_folds=10,
    random_truncations=True)

# %%
print("dumping results")
# dump results to file
with open(os.path.join('results', 'mnemosyne_mindelay.pkl'), 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

# %%

# load results from file, replacing current results
with open(os.path.join('results', 'mnemosyne_mindelay.pkl'), 'rb') as f:
    results = pickle.load(f)

# %%

stderr = lambda x: np.std(x) / np.sqrt(len(x))

# %%

plt.title('Global Difficulty')
plt.xlabel('Minimum Delay (Seconds)')
plt.ylabel('AUC')
plt.errorbar(
    min_delays,
    [np.mean(results.validation_aucs('GMIND%d' % x)) for x in min_delays],
    yerr=[stderr(results.validation_aucs('GMIND%d' % x)) for x in min_delays],
    label='Validation')
plt.scatter(
    min_delays, [results.test_auc('GMIND%d' % x) for x in min_delays],
    color='orange', linewidth=0, label='Test')
plt.xscale('log')
plt.legend(loc='best')
plt.savefig(os.path.join('figures', 'mnemosyne_mindelay',
                         'auc-vs-mindelay-global-difficulty.pdf'),
            bbox_inches='tight')
plt.show()

# %%

plt.title('Item-specific Difficulty')
plt.xlabel('Minimum Delay (Seconds)')
plt.ylabel('AUC')
plt.errorbar(
    min_delays,
    [np.mean(results.validation_aucs('GMIND%d' % x)) for x in min_delays],
    yerr=[stderr(results.validation_aucs('GMIND%d' % x)) for x in min_delays],
    label='Validation')
plt.scatter(
    min_delays, [results.test_auc('GMIND%d' % x) for x in min_delays],
    color='orange', linewidth=0, label='Test')
plt.xscale('log')
plt.legend(loc='best')
plt.savefig(os.path.join('figures', 'mnemosyne_mindelay',
                         'auc-vs-mindelay-item-difficulty.pdf'),
            bbox_inches='tight')
plt.show()

# %%

plt.xlabel('Minimum Delay (Seconds)')
plt.ylabel('Validation AUC')
plt.errorbar(
    min_delays,
    [np.mean(results.validation_aucs('GMIND%d' % x)) for x in min_delays],
    yerr=[stderr(results.validation_aucs('GMIND%d' % x)) for x in min_delays],
    label='Global Difficulty')
plt.errorbar(
    min_delays,
    [np.mean(results.validation_aucs('IMIND%d' % x)) for x in min_delays],
    yerr=[stderr(results.validation_aucs('IMIND%d' % x)) for x in min_delays],
    label='Item-specific Difficulty')
plt.xscale('log')
plt.legend(loc='best')
plt.savefig(os.path.join('figures', 'mnemosyne_mindelay',
                         'auc-vs-mindelay-global-vs-item-difficulty.pdf'),
            bbox_inches='tight')
plt.show()
