# %%

from __future__ import division

from collections import defaultdict
import json
import pickle
import os

import pandas as pd
import numpy as np
from scipy import special

# %%

from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib as mpl

mpl.rc('savefig', dpi=200)
mpl.rc('text', usetex=True)

# %%
# Load the log data from the Mechanical Turk experiments
# %%

data = []
with open(os.path.join('data', 'dataset_leitner.dump'), 'rb') as f:
    for line in f:
        data.append(json.loads(line))

# %%

num_decks = 5
session_duration = 15 * 60  # 15 minutes

# %%

std_err = lambda x: np.nanstd(x) / np.sqrt(len(x))

# %%

df = pd.DataFrame(data)

# %%

df.sort_values('card_time', inplace=True)  # sort in chronological order

# %%
# %%
# Do some pre - processing
# %%

df = df[df['worker_id'].apply(
    lambda x: 'IGOR' not in x)]  # filter out artifacts from platform debugging

# %%

df['user_id'] = df['worker_id']  # a 'user' can have multiple sessions
df['worker_id'] = df['worker_id'] + df['vocab']  # a 'worker' exists for a single session

# %%

# deck = num_decks corresponds to a new item (i.e., deck = 0) for sessions other than those for vocab.list.japanese.0
# don't need to take the deck column too seriously, since we re-compute decks for important analysis
df['deck'] = df.apply(lambda row: 0 if row['deck'] == num_decks and row[
    'vocab'] != 'vocab.list.japanese.0' else row['deck'], axis=1)
df['deck'] = df['deck'] + 1  # shift decks from [0, num_decks-1] to [1, num_decks]

# %%

nreps = []
delay = []
user_items = (df['worker_id'] + '-' + df['foreign']).unique()
nreps_of_user_item = {k: 0 for k in user_items}
prev_timestamp_of_user_item = {k: np.nan for k in user_items}
for _, ixn in df.iterrows():
    user_item = ixn['worker_id'] + '-' + ixn['foreign']
    timestamp = ixn['card_time']

    nreps.append(nreps_of_user_item[user_item])
    delay.append(timestamp - prev_timestamp_of_user_item[user_item])

    nreps_of_user_item[user_item] += 1
    prev_timestamp_of_user_item[user_item] = timestamp

# %%

df['nreps'] = nreps  # number of repetitions for user-item pair
df[
    'delay'] = delay  # time elapsed (milliseconds) since previous review for user-item pair

# %%

df['outcome'] = df['score'].apply(
    lambda x: 0 if x <= 2 else 1)  # discretize scores into binary outcomes

# %%

# extract assigned arrival rate for each interaction
arrival_rate = []
for _, row in df.iterrows():
    ar = np.nan
    if not np.isnan(row['rate']):
        ar = row['rate']
    elif type(row['probs']) == list:
        ar = row['probs'][-1]
    arrival_rate.append(ar)
df['arrival_rate'] = arrival_rate

# %%

# how much data is there for each experimental condition?
df['arrival_rate'].value_counts()
#
# # %% md
#
# Compute
# basic
# stats
# summarizing
# the
# data
#
# # %%

num_items_per_session = []
recall_rates = []
session_lengths = []
for _, group in df.groupby('worker_id'):
    num_items_per_session.append(len(group['foreign'].unique()))
    recall_rates.append(np.mean(group['outcome']))
    session_lengths.append(len(group))
#
# # %%
#
print(
"Number of interactions = %d" % len(df))
print(
"Number of users = %d" % len(df['user_id'].unique()))
print(
"Number of items = %d" % len(df['foreign'].unique()))
print(
"Number of sessions = %d" % len(df['worker_id'].unique()))
print(
"Overall recall rate = %0.3f" % np.mean(df['outcome']))
print(
"Average number of interactions in session = %0.3f" % np.mean(session_lengths))

# %%

np.mean(num_items_per_session)

# %%

plt.xlabel('Number of Unique Items Seen During Session')
plt.ylabel('Frequency (Number of Sessions)')
plt.hist(num_items_per_session)
plt.savefig(
    os.path.join('figures', 'mturk', 'num-unique-items-seen-per-session.pdf'))
plt.show()
#
# # %% md
#
# Our
# estimate
# of
# the
# empirical
# review
# frequency
# budget $U$ is as follows
#
# # %%
#
# np.mean(np.array(session_lengths) / session_duration)
#
# # %%

plt.xlabel('log10(Number of Interactions In Session)')
plt.ylabel('Frequency (Number of Sessions)')
plt.hist(np.log10(np.array(session_lengths) + 1))
plt.savefig(os.path.join('figures', 'mturk', 'num-ixns-per-session.pdf'))
plt.show()
#
# # %%

num_sessions_per_person = []
for _, group in df.groupby('user_id'):
    num_sessions_per_person.append(len(group['vocab'].unique()))

# %%
#
# np.mean(num_sessions_per_person)
#
# # %%
#
plt.xlabel('Number of Sessions')
plt.ylabel('Frequency (Number of Users)')
plt.hist(num_sessions_per_person)
plt.savefig(os.path.join('figures', 'mturk', 'num-sessions-per-person.pdf'))
plt.show()
#
# # %%
#
decks = range(1, 1 + num_decks)
outcomes = [None] * num_decks
for deck, group in df[~np.isnan(df['deck'])].groupby('deck'):
    if deck <= num_decks:
        outcomes[int(deck) - 1] = group['outcome'].values
#
# # %%
#
plt.xlabel('Deck')
plt.ylabel('Empirical Recall Rate')
plt.errorbar(decks, [np.nanmean(x) for x in outcomes],
             yerr=[std_err(x) for x in outcomes])
plt.xticks(decks)
plt.savefig(os.path.join('figures', 'mturk', 'recall-rate-vs-deck.pdf'))
plt.show()
#
# # %%
#
nreps = range(max(df['nreps']) + 1)
outcomes = [df[df['nreps'] == x]['outcome'].values for x in nreps]

# # %%

plt.xlabel('Number of repetitions')
plt.ylabel('Empirical Recall Rate')
plt.errorbar(nreps, [np.nanmean(x) for x in outcomes],
             yerr=[std_err(x) for x in outcomes])
plt.xscale('log')
plt.savefig(os.path.join('figures', 'mturk', 'recall-rate-vs-nreps.pdf'))
plt.show()
#
# # %%
#
delay_ticks = np.arange(0, 6.5, 0.1)
recall_rates = []
for x, y in zip(delay_ticks[:-1], delay_ticks[1:]):
    recall_rates.append(df[df['delay'].apply(
        lambda z: z > 0 and np.log10(1 + z) >= x and np.log10(1 + z) < y)][
                            'outcome'].values)
#
# # %%
#
plt.xlabel('log10(Delay) (log10-milliseconds)')
plt.ylabel('Empirical Recall Rate')
plt.errorbar([(x + y) / 2 for x, y in zip(delay_ticks[:-1], delay_ticks[1:])],
             [np.mean(x) if len(x) else np.nan for x in recall_rates],
             yerr=[std_err(x) if len(x) else np.nan for x in recall_rates])
plt.savefig(os.path.join('figures', 'mturk', 'recall-rate-vs-delay.pdf'))
plt.show()
#
# # %%
#
plt.xlabel('log10(Delay) (log10-milliseconds)')
plt.ylabel('Normalized Frequency (Fraction of Total Interactions)')

x = np.array(df[df['outcome'] == 0]['delay'].values)
x = x[~np.isnan(x)]
x = x[x > 0]
plt.hist(np.log10(1 + x), alpha=0.5, label='forgotten', density=True,
         linewidth=0)  # , bins=20)

x = np.array(df[df['outcome'] == 1]['delay'].values)
x = x[~np.isnan(x)]
x = x[x > 0]
plt.hist(np.log10(1 + x), alpha=0.5, label='recalled', density=True,
         linewidth=0)  # , bins=20)

plt.legend(loc='best')
plt.savefig(os.path.join('figures', 'mturk', 'delays-cond-outcomes.pdf'))
plt.show()
#
# # %%
#
fpr = []
for _, group in df.groupby('user_id'):
    vc = group['score'].value_counts()
    fpr.append(1 - ((1 + vc.get(0, 0)) / (
                2 + vc.get(0, 0) + vc.get(1, 0) + vc.get(2, 0))))

# # %%

plt.xlabel("False Positive Rate")
plt.ylabel('Frequency (Number of Users)')
plt.title('Know Thyself, Turker!')
plt.hist(fpr, bins=20)
plt.savefig(os.path.join('figures', 'mturk', 'know-thyself-fpr.pdf'))
plt.show()
#
# # %%
#
plt.xlabel('Recall Rate')
plt.ylabel('Frequency (Number of Sessions)')
plt.hist(df.groupby('worker_id')['outcome'].mean().values)  # , bins=20)
plt.savefig(os.path.join('figures', 'mturk', 'user-recall-rates.pdf'))
plt.show()
#
# # %%
#
# df.groupby('foreign')['outcome'].mean().sort(inplace=False)
#
# # %%
#
plt.xlabel('Recall Rate')
plt.ylabel('Frequency (Number of Items)')
plt.hist(df.groupby('foreign')['outcome'].mean().values)  # , bins=20)
plt.savefig(os.path.join('figures', 'mturk', 'item-recall-rates.pdf'))
plt.show()
#
# # %%
#
df_fit = df[~np.isnan(df['delay'])]
#
# # %%
#
delays = np.array(df_fit['delay'].values) / 1000  # seconds
decks = np.array(df_fit['deck'])
nreps = np.array(df_fit['nreps']) + 1
outcomes = np.array(df_fit['outcome'])
#
# # %%
#
thetas = np.arange(0.0072, 0.0082, 0.00001)
#
# # %%
#
# len(thetas)
#
# # %%
#
with np.errstate(divide='ignore', invalid='ignore'):
    lls = []
    for theta in thetas:
        ll_pass = -theta * delays / decks
        ll_fail = np.log(1 - np.exp(-theta * delays / decks))
        ll = outcomes * ll_pass + (1 - outcomes) * ll_fail
        lls.append(np.nansum(ll[np.isfinite(ll)]))
#
# # %%
#
lls = np.array(lls)
marginal_lik = special.logsumexp(lls)
#
# # %%
#
posteriors = np.exp(lls - marginal_lik)
#
# # %%
#
plt.xlabel(r'Item Difficulty $\theta$')
plt.ylabel(r'Posterior Probability $P(\theta \mid D)$')
plt.plot(thetas, posteriors)
plt.savefig(os.path.join('figures', 'mturk', 'item-difficulty-posterior.pdf'))
plt.show()
#
# # %% md
#
# Our
# maximum - likelihood
# estimate
# of
# the
# global item
# difficulty $\theta$ is as follows
#
# # %%
#
# thetas[max(range(len(thetas)), key=lambda x: lls[x])]
#
# # %% md
#
# Examine
# phase
# transition
#
# # %%
#
arrival_rate = []
final_deck_distrn = []
num_mastered = []
for worker_id, group in df.groupby('worker_id'):
    try:
        vx = int(100 * group['arrival_rate'].values[
            -1]) / 100  # handle weird rounding issues

        # re-compute the 'deck' column
        deck_of_item = {k: 0 for k in group['foreign'].unique()}
        for _, ixn in group.iterrows():
            item = ixn['foreign']
            outcome = ixn['outcome']
            if outcome == 1:
                deck_of_item[item] += 1
            elif outcome == 0 and deck_of_item[item] > 0:
                deck_of_item[item] -= 1
        items_of_deck = defaultdict(set)
        for k, v in deck_of_item.items():
            items_of_deck[min(v, num_decks)] |= {k}
        vy = [len(items_of_deck[x]) for x in range(num_decks + 1)]

        vz = vy[-1]
        arrival_rate.append(vx)
        final_deck_distrn.append(vy)
        num_mastered.append(vz)
    except Exception as e:
        print(f"I encounter error '{e}', I will pass")
        pass

# # %%
#
unique_arrival_rates = sorted(set(arrival_rate))
num_mastered_of_arrival_rate = {k: [] for k in unique_arrival_rates}
final_deck_distrn_of_arrival_rate = {k: [] for k in unique_arrival_rates}
for x, y, z in zip(arrival_rate, num_mastered, final_deck_distrn):
    num_mastered_of_arrival_rate[x].append(y)
    final_deck_distrn_of_arrival_rate[x].append(z)
#
# # %%
#
# unique_arrival_rates
#
# # %%
#
# scale arrival rates from probabilities to proper 'rates' (i.e., having units 'items per second')
scaled_unique_arrival_rates = np.array(unique_arrival_rates) * np.mean(
    session_lengths) / session_duration
#
# # %%
#
# scaled_unique_arrival_rates
#
# # %%

# with open(os.path.join('results',
#                        'theoretical-vs-simulated-phase-transition.pkl'),
#           'rb') as f:
#     simulated_arrival_rates, simulated_throughputs, theoretical_phase_transition_threshold = pickle.load(
#         f)
#
# # %%

# plt.xlabel(r'Arrival Rate $\lambda_{ext}$ (Items Per Second)')
# plt.ylabel(r'Throughput $\lambda_{out}$ (Items Per Second)')
# plt.errorbar(
#     scaled_unique_arrival_rates[:-1],
#     [np.mean(np.array(num_mastered_of_arrival_rate[x]) / (15 * 60)) for x in
#      unique_arrival_rates[:-1]],
#     yerr=[std_err(np.array(num_mastered_of_arrival_rate[x]) / (15 * 60)) for x
#           in unique_arrival_rates[:-1]],
#     label='Empirical', color='orange')
# plt.errorbar(simulated_arrival_rates,
#              [np.mean(y) for y in simulated_throughputs],
#              yerr=[std_err(y) for y in simulated_throughputs],
#              label='Simulated (Clocked Delay)', color='green')
# plt.legend(loc='best')
# plt.savefig(os.path.join('figures', 'mturk',
#                          'empirical-and-simulated-throughput-vs-arrival-rate.pdf'))
# plt.show()
#
# # %%
#
# plt.xlabel(r'Arrival Rate $\lambda_{ext}$ (Items Per Second)')
# plt.ylabel(r'Throughput $\lambda_{out}$ (Items Per Second)')
# plt.errorbar(
#     scaled_unique_arrival_rates[:-1],
#     [np.mean(np.array(num_mastered_of_arrival_rate[x]) / (15 * 60)) for x in
#      unique_arrival_rates[:-1]],
#     yerr=[std_err(np.array(num_mastered_of_arrival_rate[x]) / (15 * 60)) for x
#           in unique_arrival_rates[:-1]],
#     label='Empirical', color='orange')
# plt.errorbar(simulated_arrival_rates,
#              [np.mean(y) for y in simulated_throughputs],
#              yerr=[std_err(y) for y in simulated_throughputs],
#              label='Simulated (Clocked Delay)', color='green')
# plt.axvline(x=theoretical_phase_transition_threshold,
#             label='Phase Transition Threshold (Theoretical)', linestyle='--')
# plt.legend(loc='best')
# plt.savefig(os.path.join('figures', 'mturk',
#                          'empirical-and-simulated-and-theoretical-throughput-vs-arrival-rate.pdf'))
# plt.show()
#
# # %%
#
plt.xlabel('Arrival Rate $\lambda_{ext}$ (Items Per Second)')
plt.ylabel('Number of Items')
deck_distrns = [[[] for _ in unique_arrival_rates[:-1]] for _ in
                range(num_decks + 1)]
for i, x in enumerate(unique_arrival_rates[:-1]):
    for deck_distrn in final_deck_distrn_of_arrival_rate[x]:
        y = np.array(deck_distrn, dtype=float)
        for j, z in enumerate(y):
            deck_distrns[j][i].append(z)

for i, dd in enumerate(deck_distrns):
    label = 'Deck %d' % (i + 1)
    if i == num_decks:
        label = 'Mastered'
    plt.errorbar(scaled_unique_arrival_rates[:-1], [np.mean(x) for x in dd],
                 yerr=[std_err(x) for x in dd], label=label)
plt.legend(loc='best')
plt.savefig(os.path.join('figures', 'mturk',
                         'num-items-vs-arrival-rate-cond-deck.pdf'))
plt.show()
#
# # %%
#
plt.xlabel('Arrival Rate $\lambda_{ext}$ (Items Per Second)')
plt.ylabel('Fraction of Items Seen During Session')
deck_distrns = [[[] for _ in unique_arrival_rates[:-1]] for _ in
                range(num_decks + 1)]
for i, x in enumerate(unique_arrival_rates[:-1]):
    for deck_distrn in final_deck_distrn_of_arrival_rate[x]:
        y = np.array(deck_distrn, dtype=float)
        y /= y.sum()
        for j, z in enumerate(y):
            deck_distrns[j][i].append(z)

for i, dd in enumerate(deck_distrns):
    label = 'Deck %d' % (i + 1)
    if i == num_decks:
        label = 'Mastered'
    plt.errorbar(scaled_unique_arrival_rates[:-1], [np.mean(x) for x in dd],
                 yerr=[std_err(x) for x in dd], label=label)
plt.legend(loc='best')
plt.savefig(os.path.join('figures', 'mturk',
                         'frac-items-vs-arrival-rate-cond-deck.pdf'))
plt.show()

# # %%
#
plt.xlabel('Deck')
plt.ylabel('Fraction of Items Seen During Session')
colors = [None] * len(unique_arrival_rates[:-1])
colors[1] = 'red'
colors[3] = 'orange'
colors[7] = 'deepskyblue'
colors[10] = 'blue'
for i, (x, z) in enumerate(
        zip(unique_arrival_rates[:-1], scaled_unique_arrival_rates[:-1])):
    if i not in [1, 3, 7, 10]:  # cherry-picked
        continue
    deck_distrns = [[] for _ in range(num_decks + 1)]
    for deck_distrn in final_deck_distrn_of_arrival_rate[x]:
        y = np.array(deck_distrn, dtype=float)
        y /= y.sum()
        for j, w in enumerate(y):
            deck_distrns[j].append(w)

    plt.errorbar(
        range(1, len(deck_distrns) + 1), [np.mean(x) for x in deck_distrns],
        yerr=[std_err(x) for x in deck_distrns],
        label=r'$\lambda_{ext} = %0.3f$ (%s Phase Transition)' % (
        z, 'Before' if i <= 3 else 'After'),
        color=colors[i])
plt.xlim([0.5, num_decks + 1.5])
plt.legend(loc='best')
plt.savefig(os.path.join('figures', 'mturk',
                         'frac-items-vs-deck-cond-arrival-rate.pdf'))
plt.show()
#
# # %%
