# %%

from __future__ import division

import pickle
import os

from sklearn import metrics
import numpy as np
import pandas as pd

from lentil import evaluate
from lentil import models

import mem

# %%

from matplotlib import pyplot as plt
import seaborn as sns
# % matplotlib
# inline

# %%

import matplotlib as mpl

mpl.rc('savefig', dpi=300)
mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsfonts}')

# %%

import matplotlib.lines as mlines

# %%

import logging

logging.getLogger().setLevel(logging.DEBUG)

# %%

with open(os.path.join('data', 'dutch_big_history.pkl'), 'rb') as f:
    history = pickle.load(f)

# %% md
#
# Build
# content
# features
# for `mnemosyne_v2`

# %%

with open(os.path.join('data', 'content_features.pkl'), 'rb') as f:
    contents_of_item_id = pickle.load(f)

# %%

content_features = {k: (len(f) if f is not None else len(b)) for k, (b, f) in
                    contents_of_item_id.iteritems()}

# %% md
#
# Build
# content
# features
# for `dutch_big`

# %%

with open(os.path.join('data', 'original_of_module_id.pkl'), 'rb') as f:
    original_of_module_id = pickle.load(f)

# %%

embedding_of_word = {}
with open(os.path.join('data', 'embeddings', 'cbow', 'size=50.embeddings'),
          'rb') as f:
    for line in f:
        fields = line.strip().split(' '.encode())
        embedding_of_word[fields[0]] = np.array([float(x) for x in fields[1:]])

# %%

count_of_word = {}
with open(os.path.join('data', 'embeddings', 'cbow', 'unigram_frequencies'),
          'rb') as f:
    for line in f:
        fields = line.strip().split(' '.encode())
        count_of_word[fields[0]] = int(fields[1])
total_count = sum(count_of_word.values())
freq_of_word = {k: (v / total_count) for k, v in count_of_word.items()}

# %%

content_features = {k: np.append(
    embedding_of_word[original_of_module_id[k]],
    [len(original_of_module_id[k]), freq_of_word[original_of_module_id[k]]])
    for k in history.data['module_id'].unique()}
print(content_features)

# %%

content_features = {k: np.array(
    [len(original_of_module_id[k]), freq_of_word[original_of_module_id[k]]])
    for k in history.data['module_id'].unique()}
print(content_features)

# %%

content_features = {k: np.array(
    [len(original_of_module_id[k])]) \
    for k in history.data['module_id'].unique()}
print(content_features)

# %% md

# Setup
# the
# IRT
# benchmark
# models and memory
# models


# %%

def build_1pl_irt_model(history, filtered_history, split_history=None):
    model = models.OneParameterLogisticModel(
        filtered_history, select_regularization_constant=True,
        name_of_user_id='user_id')
    model.fit()
    return model


def build_2pl_irt_model(history, filtered_history, split_history=None):
    model = models.TwoParameterLogisticModel(
        filtered_history, select_regularization_constant=True,
        name_of_user_id='user_id')
    model.fit()
    return model


def build_student_biased_coin_model(history, filtered_history,
                                    split_history=None):
    model = models.StudentBiasedCoinModel(history, filtered_history,
                                          name_of_user_id='user_id')
    model.fit()
    return model


def build_assessment_biased_coin_model(history, filtered_history,
                                       split_history=None):
    model = models.AssessmentBiasedCoinModel(history, filtered_history)
    model.fit()
    return model


def meta_build_efc_model(
        strength_model='deck', using_delay=True,
        using_global_difficulty=True, debug_mode_on=True,
        content_features=None,
        coeffs_regularization_constant=1e-6,
        item_bias_regularization_constant=1e-6,
        using_item_bias=True):
    def build_efc_model(history, filtered_history, split_history=None):
        model = mem.EFCModel(
            filtered_history, strength_model=strength_model,
            using_delay=using_delay,
            using_global_difficulty=using_global_difficulty,
            debug_mode_on=debug_mode_on,
            content_features=content_features, using_item_bias=using_item_bias)
        model.fit(
            learning_rate=0.1,
            # learning_rate=(1 if not using_global_difficulty else 0.1),
            ftol=1e-6, max_iter=10000,
            coeffs_regularization_constant=coeffs_regularization_constant,
            item_bias_regularization_constant=item_bias_regularization_constant)
        return model

    return build_efc_model


def meta_build_logistic_regression_model(C=1.0):
    def build_logistic_regression_model(history, filtered_history,
                                        split_history=None):
        model = mem.LogisticRegressionModel(filtered_history)
        model.fit(C=C)
        return model

    return build_logistic_regression_model


# %%

model_builders = {
    '1PL IRT': build_1pl_irt_model,
    'EFC I/-/-': meta_build_efc_model(
        strength_model='deck', using_delay=True, using_global_difficulty=False,
        content_features=None, using_item_bias=True,
        item_bias_regularization_constant=1e-3),
    'EFC I/G/-': meta_build_efc_model(
        strength_model='deck', using_delay=True, using_global_difficulty=True,
        content_features=None, using_item_bias=True,
        item_bias_regularization_constant=1e-3,
        coeffs_regularization_constant=1e-3),
    'EFC I/G/B': meta_build_efc_model(
        strength_model='deck', using_delay=True, using_global_difficulty=True,
        content_features=content_features, using_item_bias=True,
        item_bias_regularization_constant=1e-3,
        coeffs_regularization_constant=1e-3),
    'EFC -/G/B': meta_build_efc_model(
        strength_model='deck', using_delay=True, using_global_difficulty=True,
        content_features=content_features, using_item_bias=False,
        coeffs_regularization_constant=1e-3),
    'EFC -/-/B': meta_build_efc_model(
        strength_model='deck', using_delay=True, using_global_difficulty=False,
        content_features=content_features, using_item_bias=False,
        coeffs_regularization_constant=1e-3)
}

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

results = evaluate.cross_validated_auc(
    model_builders,
    history,
    num_folds=10,
    random_truncations=True)

# %%

# dump results to file
with open(os.path.join('results', 'dutch_big_lesion_analysis.pkl'), 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

# %%

# load results from file, replacing current results
with open(os.path.join('results', 'dutch_big_lesion_analysis.pkl'), 'rb') as f:
    results = pickle.load(f)

# %%

df = history.data

# %% md

# Compute
# validation
# AUCs
# for separate bins of data

# %%


def compute_auc(y_trues, probas_pred):
    try:
        y_trues, probas_pred = zip(
            *[(y, p) for y, p in zip(y_trues, probas_pred) if not np.isnan(p)])
        fpr, tpr, thresholds = metrics.roc_curve(y_trues, probas_pred,
                                                 pos_label=1)
        return metrics.auc(fpr, tpr)
    except:
        return np.nan


# %%

ndata_in_logs = [df['module_id'].ix[idxes].value_counts() for
                 idxes, y_trues, probas_pred in results.train_ixn_data]
ndata_of_val_ixns = [df['module_id'].ix[idxes].apply(lambda x: vc.get(x, 0))
                     for vc, (idxes, y_trues, probas_pred) in
                     zip(ndata_in_logs, results.val_ixn_data)]

# %%

num_bins = 5
hist, bin_edges = np.histogram([y for x in ndata_of_val_ixns for y in x],
                               bins=num_bins)
t = [(x + y) / 2 for x, y in zip(bin_edges[:-1], bin_edges[1:])]

# %%

model_names = [
    '1PL IRT',
    'EFC I/-/-',
    'EFC I/G/-',
    'EFC I/G/B',
    'EFC -/G/B',
    'EFC -/-/B']

model_labels = [
    '1PL IRT',
    r'$\gamma_i$',
    r'$\gamma_i + \beta_0$',
    r'$\gamma_i + \beta_0 + \vec{\beta}_{1:n} \cdot \vec{x}_i$',
    r'$\beta_0 + \vec{\beta}_{1:n} \cdot \vec{x}_i$',
    r'$\vec{\beta}_{1:n} \cdot \vec{x}_i$']

# %%

plt.xlabel(r'$\log{(\theta_i)}$')
plt.boxplot([results.validation_aucs(m) for m in model_names])
plt.scatter(
    range(1, len(model_names) + 1),
    [results.test_auc(m) for m in model_names],
    color='orange', s=100)

plt.xticks(
    range(1, len(model_names) + 1),
    model_labels, rotation=15)
plt.xlim([0.5, len(model_names) + .5])

orange_circle = mlines.Line2D([], [], color='orange', marker='o', label='Test')
red_line = mlines.Line2D([], [], color='red', marker='_', label='Validation')
plt.legend(handles=[red_line, orange_circle], loc='best')

plt.ylabel('AUC')

plt.savefig(os.path.join('figures', 'dutch_big', 'auc-box-plots-efc-cgi.pdf'),
            bbox_inches='tight')
plt.show()

# %%

label_of_m = dict(zip(model_names, model_labels))

# %%

s_of_model = {}
for m in model_names:
    s_of_model[m] = [[compute_auc(
        [p for p, q in zip(y_trues, vf) if
         q >= x and (q < y or (bidx == len(bin_edges) - 2 and q == y))],
        [p for p, q in zip(probas_pred[m], vf) if
         q >= x and (q < y or (bidx == len(bin_edges) - 2 and q == y))]) \
        for (_, y_trues, probas_pred), vf in
        zip(results.val_ixn_data, ndata_of_val_ixns)] \
        for bidx, (x, y) in enumerate(zip(bin_edges[:-1], bin_edges[1:]))]

# %%

fig, ax1 = plt.subplots()

sns.set_style('dark')
ax2 = ax1.twinx()
ax2.bar(bin_edges[:-1], hist,
        [y - x for x, y in zip(bin_edges[:-1], bin_edges[1:])], color='gray',
        alpha=0.5, linewidth=0)
ax2.set_ylabel('Frequency (number of interactions)')

sns.set_style('darkgrid')
lines = []
for m, s1 in s_of_model.items():
    l1 = ax1.errorbar(
        t, [np.nanmean(z) for z in s1], label='%s' % label_of_m[m],
        yerr=[np.nanstd(z) / np.sqrt(len(z)) for z in s1])
    lines.append(l1)
ax1.set_xlabel('Number of training logs for item')
ax1.set_ylabel('Validation AUC')

first_legend = plt.legend(handles=lines[:3], loc='lower center',
                          bbox_to_anchor=(0.25, -0.4))
plt.gca().add_artist(first_legend)
plt.legend(handles=lines[3:], loc='lower center', bbox_to_anchor=(0.75, -0.4))

plt.savefig(os.path.join('figures', 'dutch_big', 'auc-vs-ndata.pdf'),
            bbox_inches='tight')
plt.show()

# %%