# %%

from __future__ import division

from collections import defaultdict
import pickle
import os
import sys
import copy
import random
import json

import pygraphviz as pgv
import numpy as np
import pandas as pd
import xml.etree.ElementTree

from lentil import datatools

# %%

from matplotlib import pyplot as plt
import seaborn as sns
# % matplotlib
# inline

# %%

import matplotlib as mpl

mpl.rc('savefig', dpi=300)
mpl.rc('text', usetex=True)

# %% md
# Download the Mnemosyne log data from [here](https://archive.org/details/20140127MnemosynelogsAll.db) and add the decompressed contents to the `data` directory. I would recommend creating a reasonably-sized random sample of logs from the full db before loading the data into this notebook, since there are ~120 million logs in total. You can use the following commands:
#
# ```
# sqlite3 2014-01-27-mnemosynelogs-all.db
# .mode csv
# .headers on
# .output mnemosynelogs_mini.csv
# select * from log order by Random() limit 10000000;

# %%

public_itemids = defaultdict(set)
fs = [x for x in os.listdir(os.path.join('data', 'shared_decks')) if
      '.xml' in x]
for f in fs:
    try:
        e = xml.etree.ElementTree.parse(
            os.path.join('data', 'shared_decks', f)).getroot()
        for x in e.findall('log'):
            public_itemids[x.get('o_id')].add(f)
    except:
        continue

# %%

len(public_itemids)

# %%

num_logs_of_file = defaultdict(int)
num_logs_of_item = defaultdict(int)
logged_itemids = set()
num_public_logs = 0
with open(os.path.join('data', 'mnemosynelogs_itemids_full.csv'), 'r') as f:
    f.readline()
    for line in f:
        line = line.replace('\r\n', '')
        if line != '':
            if line in public_itemids:
                num_public_logs += 1
                num_logs_of_item[line] += 1
                for f in public_itemids[line]:
                    num_logs_of_file[f] += 1
            logged_itemids.add(line)

# %%

print(num_public_logs)

# %%

print(len(logged_itemids))

# %%

print(sum(1 for x in public_itemids if x in logged_itemids))

# %%

print(sorted(num_logs_of_item.items(), key=lambda items: items[1], reverse=True)[:500])

# %%

print(sorted(num_logs_of_file.items(), key=lambda items: items[1], reverse=True)[:50])


# %%

def contents_of_items_in_file(f):
    e = xml.etree.ElementTree.parse(
        os.path.join('data', 'shared_decks', f)).getroot()
    D = {}
    M = {}
    for x in e.findall('log'):
        if x.get('type') == '16':
            b = x.find('b')
            if b is None:
                b = x.find('m_1')
            f = x.find('f')
            if b is not None or f is not None:
                D[x.get('o_id')] = (b.text if b is not None else None,
                                    f.text if f is not None else None)
        elif x.get('type') == '6':
            M[x.get('o_id')] = x.get('fact')
    return {k: D[v] for k, v in M.items()}


# %%

contents_of_item_id = {}
for f in os.listdir(os.path.join('data', 'shared_decks')):
    if '.xml' in f:
        try:
            contents_of_item_id.update(contents_of_items_in_file(f))
        except:
            pass

# %%

print(len(contents_of_item_id))

# %%

print(contents_of_item_id)

# %%

with open(os.path.join('data', 'content_features.pkl'), 'wb') as f:
    pickle.dump(contents_of_item_id, f, pickle.HIGHEST_PROTOCOL)

# %% md

# Filter log for public items

# %%

with open(os.path.join('data', 'mnemosynelogs_full.csv'), 'rb') as f:
    with open(os.path.join('data', 'mnemosynelogs_full_filtered.csv'),
              'wb') as g:
        g.write(f.readline())
        for line in f:
            fields = line.split(','.encode())
            if fields[4] != '' and fields[3] in contents_of_item_id:
                g.write(line)

# %% md

# Make
# the
# data
# set
# manageably
# smaller
# by
# filtering
# out
# users
# with short / long review histories

# %%

unfiltered_logs = pd.read_table(
    os.path.join('data', 'mnemosynelogs_full_filtered.csv'), delimiter=',')

# %%

num_ixns_of_user = unfiltered_logs['user_id'].value_counts()

# %%

user_ids = unfiltered_logs['user_id'].unique()

# %%

mn = 10
mx = 50000
len(user_ids), sum(1 for x in user_ids if
                   num_ixns_of_user[x] > mn and num_ixns_of_user[x] < mx), sum(
    num_ixns_of_user[x] for x in user_ids if
    num_ixns_of_user[x] > mn and num_ixns_of_user[x] < mx)

# %%

user_ids = {x for x in user_ids if
            num_ixns_of_user[x] > mn and num_ixns_of_user[x] < mx}

# %%

filtered_logs = unfiltered_logs[unfiltered_logs['user_id'].isin(user_ids)]

# %%

filtered_logs.to_csv(
    os.path.join('data', 'mnemosynelogs_full_filtered_pruned.csv'),
    index=False)

# %% md
#
# Load
# the
# filtered
# logs and compute
# basic
# stats
# summarizing
# the
# data
# set

# %%

df = pd.read_csv(
    os.path.join('data', 'mnemosynelogs_full_filtered_pruned.csv'),
    delimiter=',')

# %%

print(
'\n'.join(df.columns))

# %%

len(df[~np.isnan(df['grade'])])

# %%

print(
"Number of interactions = %d" % len(df))
print(
"Number of unique students = %d" % len(df['user_id'].unique()))
print(
"Number of unique modules = %d" % len(df['object_id'].unique()))

# %%

av = np.array(df['actual_interval'].values)
sv = np.array(df['scheduled_interval'].values)
av, sv = zip(*[(x, y) for x, y in zip(av, sv) if
               x > 0 and y > 0 and not np.isnan(x) and not np.isnan(y)])

# %%

av = np.array(av)
sv = np.array(sv)

# %%

plt.xlabel('log10(Scheduled interval) (log10-milliseconds)')
plt.ylabel('Frequency (number of interactions)')
plt.hist(np.log10(sv + 1), bins=20)
plt.show()

# %%

plt.xlabel('log10(Scheduled interval) (log10-milliseconds)')
plt.ylabel('log10(Actual interval) (log10-milliseconds)')
plt.scatter(np.log10(sv + 1), np.log10(av + 1), alpha=0.005)
# plt.savefig(os.path.join('figures', 'mnemosyne', 'scheduled-vs-actual-intervals.pdf'))
plt.show()

# %%

v = np.array(df['user_id'].value_counts().values)

plt.xlabel('log10(Number of interactions per student)')
plt.ylabel('Frequency (number of students)')
plt.hist(np.log10(v))
plt.show()

# %%

v = np.array(df['object_id'].value_counts().values)

plt.xlabel('log10(Number of interactions per problem)')
plt.ylabel('Frequency (number of problems)')
plt.hist(np.log10(v))
plt.show()

# %%

grades = np.array(df['grade'].values)

plt.xlabel('Grade')
plt.ylabel('Frequency (number of interactions)')
plt.hist(grades[~np.isnan(grades)])
plt.show()

# %% md

# Apply
# more
# filters and format
# the
# log
# data
# into
# an
# `InteractionHistory`
# that
# can
# be
# understood
# by[lentil](https: // github.com / rddy / lentil)

# %%

def interaction_history_from_mnemosyne_data_set(data):
    """
 Parse Mnemosyne data set into an interaction history

 :param pd.DataFrame data: A dataframe of raw log data
 :rtype: datatools.InteractionHistory
 :return: An interaction history object
 """

    data = data[data['grade'].apply(lambda x: not np.isnan(x))]

    data = data[['user_id', 'student_id', 'object_id', 'grade', 'timestamp',
                 'thinking_time', 'actual_interval', 'scheduled_interval']]
    data.columns = ['user_id', 'student_id', 'module_id', 'outcome',
                    'timestamp', 'duration', 'actual_interval',
                    'scheduled_interval']

    data['outcome'] = data['outcome'].apply(lambda x: x > 1)

    student_timesteps = defaultdict(int)
    timesteps = [None] * len(data)
    for i, (_, ixn) in enumerate(data.iterrows()):
        student_timesteps[ixn['student_id']] += 1
        timesteps[i] = student_timesteps[ixn['student_id']]
    data['timestep'] = timesteps

    data['module_type'] = [datatools.AssessmentInteraction.MODULETYPE] * len(
        data)

    return datatools.InteractionHistory(data, sort_by_timestep=True)


# %%

df.sort('timestamp', inplace=True)

# %%

# this is helpful for splitting histories by user-item pair (instead of by user) in lentil.evaluate
df['student_id'] = [str(x['user_id']) + '-' + str(x['object_id']) for _, x in
                    df.iterrows()]

# %%

unfiltered_history = interaction_history_from_mnemosyne_data_set(df)

# %%

unfiltered_history.data['outcome'].value_counts()

# %% md

# Perform
# analagous
# preprocessing
# steps
# for the MTurk data set

# %%

data = []
with open(os.path.join('data', 'first_mturk_experiment.dataset'), 'rb') as f:
    for line in f:
        data.append(json.loads(line))

# %%

df = pd.DataFrame(data)

# %%

df['delta_t'] = df['delta_t'] * 4 * 60 * 60  # seconds

# %%

num_ixns_per_user_item = {k: defaultdict(list) for k in df['user'].unique()}
for _, ixn in df.iterrows():
    num_ixns_per_user_item[ixn['user']][ixn['item']].append(ixn['delta_t'])

# %%

start_time_of_user_item = {}
for user, num_ixns_per_item in num_ixns_per_user_item.items():
    start_time = 0
    for item, delta_ts in num_ixns_per_item.iteritems():
        start_time_of_user_item[(user, item)] = start_time
        start_time += sum(delta_ts)

# %%

df.sort('n_reps', inplace=True)

# %%

timestamps = []
for _, ixn in df.iterrows():
    user_item = (ixn['user'], ixn['item'])
    start_time_of_user_item[user_item] += ixn['delta_t']
    timestamps.append(start_time_of_user_item[user_item])

# %%

df['timestamp'] = timestamps
df.sort('timestamp', inplace=True)


# %%

def interaction_history_from_mturk_data_set(data):
    """
 Parse MTurk data set into an interaction history

 :param pd.DataFrame data: A dataframe of raw log data
 :rtype: datatools.InteractionHistory
 :return: An interaction history object
 """

    data = data[['user', 'user', 'item', 'bin_score', 'timestamp']]
    data.columns = ['user_id', 'student_id', 'module_id', 'outcome',
                    'timestamp']

    data['outcome'] = data['outcome'].apply(lambda x: x == 1)

    student_timesteps = defaultdict(int)
    timesteps = [None] * len(data)
    for i, (_, ixn) in enumerate(data.iterrows()):
        student_timesteps[ixn['student_id']] += 1
        timesteps[i] = student_timesteps[ixn['student_id']]
    data['timestep'] = timesteps

    data['module_type'] = [datatools.AssessmentInteraction.MODULETYPE] * len(
        data)

    return datatools.InteractionHistory(data, sort_by_timestep=True)


# %%

unfiltered_history = interaction_history_from_mturk_data_set(df)

# %% md

# Pre - process
# the
# `dutch_big`
# dataset

# %%

data = []
with open(os.path.join('data', 'dutch_big.dump'), 'rb') as f:
    for line in f:
        data.append((line.split('\t')[0], json.loads(line.split('\t')[1])))

# %%

original_of_module_id = {}
for _, h in data:
    for x in h:
        original_of_module_id[x['foreign']] = x['original']

# %%

with open(os.path.join('data', 'original_of_module_id.pkl'), 'wb') as f:
    pickle.dump(original_of_module_id, f, pickle.HIGHEST_PROTOCOL)

# %%

ixns = []
timestamp_of_student = defaultdict(int)
for student_id, h in data:
    for ixn in h:
        timestamp_of_student[student_id] += 1
        ixns.append(
            {'student_id': student_id, 'module_id': ixn['foreign'],
             'outcome': ixn['score'] > 2,
             'timestamp': timestamp_of_student[student_id]})

# %%

df = pd.DataFrame(ixns)

# %%

df['user_id'] = df['student_id']

# %%

df['student_id'] = df['user_id'] + '-' + df['module_id']

# %%

len(df)

# %%

df.sort('timestamp', inplace=True)


# %%

def interaction_history_from_dutch_big_data_set(data):
    """
 Parse MTurk data set into an interaction history

 :param pd.DataFrame data: A dataframe of raw log data
 :rtype: datatools.InteractionHistory
 :return: An interaction history object
 """

    data = data[['user_id', 'student_id', 'module_id', 'outcome', 'timestamp']]
    data.columns = ['user_id', 'student_id', 'module_id', 'outcome',
                    'timestamp']

    student_timesteps = defaultdict(int)
    timesteps = [None] * len(data)
    for i, (_, ixn) in enumerate(data.iterrows()):
        student_timesteps[ixn['student_id']] += 1
        timesteps[i] = student_timesteps[ixn['student_id']]
    data['timestep'] = timesteps

    data['module_type'] = [datatools.AssessmentInteraction.MODULETYPE] * len(
        data)

    return datatools.InteractionHistory(data, sort_by_timestep=True)


# %%

unfiltered_history = interaction_history_from_dutch_big_data_set(df)

# %% md

# Apply
# additional
# data
# filters


# %%

def filter_history(history, min_num_ixns=5, max_num_ixns=9223372036854775807):
    """
 Filter history for students with histories of bounded length,
 and modules with enough interactions

 :param datatools.InteractionHistory history: An interaction history
 :param int min_num_ixns: Minimum number of timesteps in student history,
     and minimum number of interactions for module

 :param int max_num_ixns: Maximum number of timesteps in student history
 :rtype: datatools.InteractionHistory
 :return: A filtered interaction history
 """
    students = set(history.data['student_id'][(
                                                      history.data[
                                                          'timestep'] > min_num_ixns) & (
                                                      history.data[
                                                          'module_type'] == datatools.AssessmentInteraction.MODULETYPE)])
    students -= set(
        history.data['student_id'][history.data['timestep'] >= max_num_ixns])

    modules = {module_id for module_id, group in
               history.data.groupby('module_id') if len(group) > min_num_ixns}

    return datatools.InteractionHistory(
        history.data[(history.data['student_id'].isin(students)) & (
            history.data['module_id'].isin(modules))],
        reindex_timesteps=True,
        size_of_test_set=0.2)


# %%

# apply the filter a couple of times, since removing student histories
# may cause certain modules to drop below the min_num_ixns threshold,
# and removing modules may cause student histories to drop below
# the min_num_ixns threshold
REPEATED_FILTER = 3  # number of times to repeat filtering
import functools
history = functools.reduce(
    lambda acc, _: filter_history(acc, min_num_ixns=2, max_num_ixns=10000),
    range(REPEATED_FILTER), unfiltered_history)

# %%

history.data.sort('timestamp', inplace=True)

# %%

deck_of_student_item = {}
tlast_of_student_item = {}
nreps_of_student_item = {}

deck = []
tlast = []
nreps = []
for _, ixn in history.data.iterrows():
    student_item = (ixn['user_id'], ixn['module_id'])
    d = deck_of_student_item.get(student_item, 1)
    deck.append(d)

    if ixn['outcome']:
        deck_of_student_item[student_item] = d + 1
    else:
        deck_of_student_item[student_item] = max(1, d - 1)

    n = nreps_of_student_item.get(student_item, 1)
    nreps.append(n)
    nreps_of_student_item[student_item] = n + 1

    tlast.append(tlast_of_student_item.get(student_item, np.nan))
    tlast_of_student_item[student_item] = ixn['timestamp']

history.data['deck'] = deck
history.data['nreps'] = nreps
history.data['tlast'] = tlast

# %%

# path to pickled interaction history file
history_path = os.path.join('data', 'mnemosyne_history_v2.pkl')

# %%

# serialize history
with open(history_path, 'wb') as f:
    pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

# %% md

# Explore
# basic
# stats
# about
# filtered, formatted
# interaction
# history

# %%

# load history from file
with open(history_path, 'rb') as f:
    history = pickle.load(f)

# %%

df = history.data

# %%

print(
"Number of interactions = %d" % len(df))
print(
"Number of unique students: %d" % len(df['user_id'].unique()))
print(
"Number of unique assessments: %d" % history.num_assessments())
value_counts = df['outcome'].value_counts()
num_passes = value_counts.get(True, 0)
num_fails = value_counts.get(False, 0)
print(
"Overall pass rate: %f" % (num_passes / (num_passes + num_fails)))

# %%

df.sort('timestamp', inplace=True)

# %%

v = []
for _, g in df.groupby(['user_id', 'module_id']):
    ts = g['timestamp'].values
    v.extend([nt - t for t, nt in zip(ts[:-1], ts[1:])])

# %%

v = np.array(v)

# %%

plt.xlabel('Time between reviews (log10-seconds)')
plt.ylabel('Frequency (number of reviews)')
plt.hist(np.log10(v + 1), bins=20)
# plt.savefig(os.path.join('figures', 'mnemosyne', 'time-between-reviews.pdf'))
plt.show()

# %%

grouped = df.groupby(['user_id', 'module_id'])

# %%

pairs = [x for x, g in grouped if len(g) > 20]

# %%

len(pairs)

# %%

g = grouped.get_group(random.choice(pairs))

# %%

ts = g['timestamp'].values
intervals = [y - x for x, y in zip(ts[:-1], ts[1:])]

plt.xlabel('Number of reviews')
plt.ylabel('Time until next review (seconds)')
plt.title('Review intervals for a single user-item pair')

outcomes = g['outcome'].values
outcomes = outcomes[:-1]
plt.bar(range(len(outcomes)), [max(intervals)] * len(outcomes), width=1,
        color=['green' if x else 'red' for x in outcomes], alpha=0.25,
        linewidth=0.)

plt.step(range(len(intervals) + 1), intervals + [intervals[-1]], where='post')

plt.yscale('log')
plt.xlim([0, len(intervals)])
plt.ylim([0, max(intervals)])

# plt.savefig(os.path.join('figures', 'mnemosyne', 'review-history-example.pdf'))
plt.show()

# %%

counts = df['user_id'].value_counts().values
plt.xlabel('Number of interactions per student')
plt.ylabel('Frequency (number of students)')
plt.hist(counts)
plt.yscale('log')
# plt.savefig(os.path.join('figures', 'mnemosyne', 'num_ixns_per_student.pdf'))
plt.show()

# %%

counts = df['module_id'][df[
                             'module_type'] == datatools.AssessmentInteraction.MODULETYPE].value_counts().values

plt.xlabel('Number of interactions per item')
plt.ylabel('Frequency (number of items)')
plt.hist(counts)
plt.yscale('log')
# plt.savefig(os.path.join('figures', 'mnemosyne', 'num_ixns_per_item.pdf'))
plt.show()

# %%

counts = df.groupby(['user_id', 'module_id']).size().values

plt.xlabel('Number of interactions per student per item')
plt.ylabel('Frequency (number of student-item pairs)')
plt.hist(counts)
plt.yscale('log')
# plt.savefig(os.path.join('figures', 'mnemosyne', 'num_ixns_per_student_per_item.pdf'))
plt.show()

# %%

num_students_per_module = [len(group['user_id'].unique()) for _, group in
                           df.groupby('module_id')]

# %%

plt.xlabel('Number of students per item')
plt.ylabel('Frequency (number of items)')
plt.hist(num_students_per_module)
plt.yscale('log')
# plt.savefig(os.path.join('figures', 'mnemosyne', 'num-students-per-item.pdf'))
plt.show()


# %%

def get_pass_rates(grouped):
    """
 Get pass rate for each group

 :param pd.GroupBy grouped: A grouped dataframe
 :rtype: dict[str, float]
 :return: A dictionary mapping group name to pass rate
 """
    pass_rates = {}
    for name, group in grouped:
        vc = group['outcome'].value_counts()
        if True not in vc:
            pass_rates[name] = 0
        else:
            pass_rates[name] = vc[True] / len(group)
    return pass_rates


# %%

grouped = df[
    df['module_type'] == datatools.AssessmentInteraction.MODULETYPE].groupby(
    'user_id')

plt.xlabel('Student pass rate')
plt.ylabel('Frequency (number of students)')
plt.hist(get_pass_rates(grouped).values())
plt.yscale('log')
# plt.savefig(os.path.join('figures', 'mnemosyne', 'student-pass-rates.pdf'))
plt.show()

# %%

grouped = df[
    df['module_type'] == datatools.AssessmentInteraction.MODULETYPE].groupby(
    'module_id')

plt.xlabel('Assessment pass rate')
plt.ylabel('Frequency (number of assessments)')
plt.hist(get_pass_rates(grouped).values())
plt.yscale('log')
# plt.savefig(os.path.join('figures', 'mnemosyne', 'assessment-pass-rates.pdf'))
plt.show()


# %%

def make_flow_graph(interaction_logs):
    """
 Create a graphviz object for the graph of
 module transitions across all student paths

 :param pd.DataFrame interaction_logs: An interaction history
 :rtype pgv.AGraph
 :return Graph of module transitions in student paths
 """
    G = pgv.AGraph(directed=True)

    for module_id in interaction_logs['module_id'].unique():
        G.add_node(module_id)

    E = defaultdict(set)
    grouped = interaction_logs.groupby('user_id')
    for student_id, group in grouped:
        module_ids_in_student_path = group['module_id']
        for source_node, target_node in zip(module_ids_in_student_path[:-1],
                                            module_ids_in_student_path[1:]):
            if source_node != target_node:  # stationary
                E[(source_node, target_node)] |= {student_id}

    for (source_node,
         target_node), students_that_made_transition in E.items():
        G.add_edge(
            source_node,
            target_node,
            weight=len(students_that_made_transition))

    return G


# %%

G = make_flow_graph(df)

# %%

G.write(os.path.join('figures', 'mnemosyne', 'mnemosyne_flow_graph.dot'))

# %%


