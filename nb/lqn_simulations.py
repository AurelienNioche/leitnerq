# %%

from __future__ import division

from collections import OrderedDict
import time
import copy
import pickle
import os
import random

import pandas as pd
import numpy as np

import sched

# %%

from matplotlib import pyplot as plt
import seaborn as sns
# % matplotlib
# inline

# %%

import matplotlib as mpl

mpl.rc('savefig', dpi=300)
mpl.rc('text', usetex=True)


# %%

def sample_arrival_times(all_items, arrival_rate, start_time):
    """
    Sample item arrival times for init_data['arrival_time_of_item'],
    which gets passed to the StatelessLQNScheduler constructor

    :param set[str] all_items: A set of item ids
    :param float arrival_rate: The arrival rate for the Poisson process
    :param int start_time: Start time (unix epoch) for the arrival process
    """

    all_items = list(all_items)
    random.shuffle(all_items)
    inter_arrival_times = np.random.exponential(1 / arrival_rate,
                                                len(all_items))
    arrival_times = start_time + np.cumsum(inter_arrival_times, axis=0).astype(
        int)
    return OrderedDict(zip(all_items, arrival_times))


# %% md
#
# Sanity
# check

# %%

init_data = {
    'arrival_time_of_item': {0: int(time.time())},
    'review_rates': np.array([0.25, 0.25, 0.25, 0.25])[np.newaxis, :],
    'difficulty_of_item': {0: 0.01},
    'difficulty_rate': 100,
    'max_num_items_in_deck': None
}

scheduler = sched.ExtLQNScheduler(init_data)

history = []

assert scheduler.next_item() == 0

# %% md
#
# Simulations

# %%

global_item_difficulty = 0.0076899999999998905
difficulty_rate = 1 / global_item_difficulty
using_global_difficulty = True

# %%

num_items = 50
difficulty_of_item = np.ones(
    num_items) * global_item_difficulty if using_global_difficulty else np.random.exponential(
    1 / difficulty_rate, num_items)

# %%

arrival_rate = 0.05
num_timesteps_in_sim = 1000

# %%

all_items = range(num_items)
start_time = int(time.time())
init_data = {
    'arrival_time_of_item': sample_arrival_times(all_items, arrival_rate,
                                                 start_time),
    'review_rates': np.array(
        [[0.125, 0.125, 0.125, 0.125], [0.125, 0.125, 0.125, 0.125]]),
    'difficulty_of_item': {i: x for i, x in enumerate(difficulty_of_item)},
    'difficulty_rate': difficulty_rate,
    'max_num_items_in_deck': None
}
scheduler = sched.ExtLQNScheduler(init_data)

# %%

num_systems = len(init_data['review_rates'])
num_decks = len(init_data['review_rates'][0])

# %%

work_rate = 0.19020740740740741
inter_arrival_times = np.random.exponential(1 / work_rate,
                                            num_timesteps_in_sim)
timesteps = int(time.time()) + np.cumsum(inter_arrival_times, axis=0).astype(
    int)

# %%

history = []

deck_of_item = {item: 1 for item in all_items}
latest_timestamp_of_item = {item: 0 for item in all_items}

for current_time in timesteps:
    try:
        next_item = scheduler.next_item(current_time=current_time)
    except sched.ExhaustedError:
        continue

    delay = current_time - latest_timestamp_of_item[next_item]
    latest_timestamp_of_item[next_item] = current_time

    deck = deck_of_item[next_item]
    outcome = 1 if np.random.random() < np.exp(
        -difficulty_of_item[next_item] * delay / deck) else 0

    deck_of_item[next_item] = max(1, deck + 2 * outcome - 1)

    history.append(
        {'item_id': next_item, 'outcome': outcome, 'timestamp': current_time})
    scheduler.update(next_item, outcome, current_time)

# %%

df = pd.DataFrame(history)

# %%

np.mean(df['outcome'])


# %%

def deck_promotion_rates(init_data, history):
    """
    Compute the observed rates at which items move from deck i to deck i+1

    :param pd.DataFrame history: The logs for a single user
    :rtype: list[float]
    :return: The average promotion rate (items per second) for each deck
    """

    deck_of_item = {item: 1 for item in init_data['arrival_time_of_item']}
    num_decks = len(init_data['review_rates'][0])
    num_promotions_of_deck = {deck: 0 for deck in range(1, num_decks + 1)}

    for ixn in history:
        item = ixn['item_id']
        outcome = ixn['outcome']
        current_deck = deck_of_item[item]
        if outcome == 1:
            if current_deck >= 1 and current_deck <= num_decks:
                num_promotions_of_deck[current_deck] += 1
            deck_of_item[item] += 1
        elif outcome == 0 and current_deck > 1:
            deck_of_item[item] -= 1

    duration = max(ixn['timestamp'] for ixn in history) - min(
        ixn['timestamp'] for ixn in history)
    promotion_rate_of_deck = {deck: (num_promotions / (1 + duration)) for
                              deck, num_promotions in
                              num_promotions_of_deck.items()}
    return promotion_rate_of_deck


# %%

deck_promotion_rates(init_data, history)

# %%

max_num_items_in_deck = None


def run_sim(arrival_rate, num_items, difficulty_rate, difficulty_of_item,
            review_rates, work_rate, num_timesteps_in_sim,
            expected_recall_likelihoods=None):
    all_items = range(num_items)
    start_time = int(time.time())
    init_data = {
        'arrival_time_of_item': sample_arrival_times(all_items, arrival_rate,
                                                     start_time),
        'review_rates': review_rates,
        'difficulty_of_item': {i: x for i, x in enumerate(difficulty_of_item)},
        'difficulty_rate': difficulty_rate,
        'max_num_items_in_deck': max_num_items_in_deck
    }
    num_decks = len(init_data['review_rates'][0])

    scheduler = sched.ExtLQNScheduler(init_data)

    history = []
    deck_of_item = {item: 1 for item in all_items}
    latest_timestamp_of_item = {item: 0 for item in all_items}

    inter_arrival_times = np.random.exponential(1 / work_rate,
                                                num_timesteps_in_sim)
    timesteps = int(time.time()) + np.cumsum(inter_arrival_times,
                                             axis=0).astype(int)
    for current_time in timesteps:
        try:
            next_item = scheduler.next_item(current_time=current_time)
        except sched.ExhaustedError:
            continue

        deck = deck_of_item[next_item]

        if expected_recall_likelihoods is None:
            delay = current_time - latest_timestamp_of_item[next_item]
            recall_likelihood = np.exp(
                -difficulty_of_item[next_item] * delay / deck)
        else:
            recall_likelihood = expected_recall_likelihoods[deck - 1]

        outcome = 1 if np.random.random() < recall_likelihood else 0

        latest_timestamp_of_item[next_item] = current_time
        deck_of_item[next_item] = max(1, deck + 2 * outcome - 1)

        history.append({'item_id': next_item, 'outcome': outcome,
                        'timestamp': current_time})
        scheduler.update(next_item, outcome, current_time)

    if history == []:
        return 0
    promotion_rate_of_deck = deck_promotion_rates(init_data, history)
    return promotion_rate_of_deck[num_decks]


# %%

num_sim_repeats = 10
num_systems = 1
num_decks = 5
work_rate = 0.19020740740740741
num_timesteps_in_sim = 500

# %%

review_rates = 1 / np.sqrt(np.arange(1, num_decks + 1, 1))
review_rates /= review_rates.sum()
review_rates = review_rates[np.newaxis, :]

# %%

run_sim(1., num_items, difficulty_rate, difficulty_of_item, review_rates,
        work_rate, num_timesteps_in_sim)

# %%

std_err = lambda x: np.nanstd(x) / np.sqrt(len(x))

# %% md
#
# Compared
# simulations
# with clocked delay to simulations with the mean-recall approximation

# %%

arrival_rates = np.arange(0.001, 0.01 + 1e-6, 0.0005)

# %%

# from lqn_properties.ipynb
expected_recall_likelihoods = [
    [0.8816669326889862, 0.912726114951097, 0.92719714973377,
     0.9360503409439428, 0.9423109652525123],
    [0.8802159353708854, 0.9112980491805251, 0.9257778382415257,
     0.9346388744415259, 0.94097809417084],
    [0.8787197882984757, 0.9098132332840261, 0.9242928082947981,
     0.9331551225315882, 0.9395782276183158],
    [0.8771758043736176, 0.9082675509248022, 0.9227366399542867,
     0.9315926867387265, 0.9381058586147527],
    [0.8755810321617427, 0.9066564105056785, 0.9211032176252993,
     0.9299443133327762, 0.9365548225241448],
    [0.8739322170539264, 0.9049746654250952, 0.9193856028066731,
     0.9282017344908691, 0.9349181903609529],
    [0.8722257544837021, 0.9032165160944434, 0.9175758756485984,
     0.9263554706857485, 0.9331881396027241],
    [0.8704576330412577, 0.9013753882808239, 0.915664935506246,
     0.9243945823202996, 0.9313557965386431],
    [0.8686233645757845, 0.8994437802941111, 0.9136422468301503,
     0.9223063540631219, 0.9294110422347164],
    [0.8667178972966402, 0.8974130685424716, 0.9114955110493042,
     0.9200758886676667, 0.9273422714786288],
    [0.864735506299407, 0.8952732565009774, 0.909210236521236,
     0.9176855771145564, 0.9251360902192634],
    [0.8626696535627787, 0.8930126452763714, 0.9067691653630414,
     0.91511439677798, 0.9227769314748718],
    [0.8605128057906617, 0.8906173931562718, 0.9041514949450417,
     0.9123369657072056, 0.9202465615555917],
    [0.858256192636374, 0.8880709140279719, 0.9013317974714279,
     0.9093222432855949, 0.9175234362738623],
    [0.8558894786472526, 0.8853530375953804, 0.8982784973778067,
     0.906031726907586, 0.9145818567364079],
    [0.8534003061323808, 0.8824387858876376, 0.8949515522602463,
     0.9024167152103361, 0.9113907773426282],
    [0.8507736192636972, 0.8792965769000917, 0.8912991768515758,
     0.8984146080316402, 0.9079123035650984],
    [0.8479907015893672, 0.8758854112910051, 0.8872523574499168,
     0.8939427107145428, 0.9040993839676039],
    [0.8450275725143461, 0.8721502713471428, 0.8827160178057104,
     0.8888886821757709, 0.8998926094017514]]

# %%

assert len(expected_recall_likelihoods) == len(arrival_rates)

# %%

ys = [[run_sim(x, num_items, difficulty_rate, difficulty_of_item, review_rates,
               work_rate - x, num_timesteps_in_sim) for _ in
       range(num_sim_repeats)] for x in arrival_rates]

# %%

exp_ys = [[run_sim(x, num_items, difficulty_rate, difficulty_of_item,
                   review_rates, work_rate - x, num_timesteps_in_sim,
                   expected_recall_likelihoods=y) for _ in
           range(num_sim_repeats)] for x, y in
          zip(arrival_rates, expected_recall_likelihoods)]

# %%

mean_ys = [np.mean(y) for y in ys]
std_err_ys = [std_err(y) for y in ys]
mean_exp_ys = [np.mean(y) for y in exp_ys]
std_err_exp_ys = [std_err(y) for y in exp_ys]

# %%

plt.xlabel(r'Arrival Rate $\lambda_{ext}$ (Items Per Second)')
plt.ylabel(r'Throughput $\lambda_{out}$ (Items Per Second)')
plt.errorbar(arrival_rates, mean_exp_ys, yerr=std_err_exp_ys,
             label='Simulated (Mean-Recall Approximation)')
plt.errorbar(arrival_rates, mean_ys, yerr=std_err_ys,
             label='Simulated (Clocked Delay)')
plt.plot(np.arange(arrival_rates[0], arrival_rates[-1], 0.0001),
         np.arange(arrival_rates[0], arrival_rates[-1], 0.0001), '--',
         label='Theoretical Steady-State Behavior')
plt.legend(loc='best')
plt.savefig(os.path.join('figures', 'lqn', 'clocked-vs-expected-delays.pdf'))
plt.show()

# %%

with open(os.path.join('results', 'clocked-vs-expected-delays.pkl'),
          'wb') as f:
    pickle.dump((arrival_rates, ys, exp_ys), f, pickle.HIGHEST_PROTOCOL)

# %% md

# Compare
# theoretical
# phase
# transition
# threshold
# to
# simulations

# %%

arrival_rates = np.arange(0.001, 0.15, 0.005)

# %%

theoretical_phase_transition_threshold = 0.013526062011718753  # from lqn_properties.ipynb

# %%

ys = [[run_sim(x, num_items, difficulty_rate, difficulty_of_item, review_rates,
               work_rate - x, num_timesteps_in_sim) for _ in
       range(num_sim_repeats)] for x in arrival_rates]

# %%

plt.xlabel(r'Arrival Rate $\lambda_{ext}$ (Items Per Second)')
plt.ylabel(r'Throughput $\lambda_{out}$ (Items Per Second)')
plt.errorbar(arrival_rates, [np.mean(y) for y in ys],
             yerr=[std_err(y) for y in ys], label='Simulated (Clocked Delay)')
plt.axvline(x=theoretical_phase_transition_threshold,
            label=r'Phase Transition Threshold (Theoretical)', linestyle='--')
plt.legend(loc='best')
plt.savefig(os.path.join('figures', 'lqn',
                         'theoretical-vs-simulated-phase-transition.pdf'))
plt.show()

# %%

with open(os.path.join('results',
                       'theoretical-vs-simulated-phase-transition.pkl'),
          'wb') as f:
    pickle.dump((arrival_rates, ys, theoretical_phase_transition_threshold), f,
                pickle.HIGHEST_PROTOCOL)

# %% md

# Compare
# simulations
# of
# different
# lengths(i.e., transient
# vs.steady - state
# behavior)

# %%

arrival_rates = np.arange(0.001, 0.15, 0.0001)
sim_lengths = [500, 1000, 5000, 10000]

# %%

num_items = 500
difficulty_of_item = np.ones(
    num_items) * global_item_difficulty if using_global_difficulty else np.random.exponential(
    global_item_difficulty, num_items)

# %%

ys = [[[run_sim(x, num_items, difficulty_rate, difficulty_of_item,
                review_rates, work_rate - x, y) for _ in
        range(num_sim_repeats)] for x in arrival_rates] for y in sim_lengths]

# %%

plt.xlabel(r'Arrival Rate $\lambda_{ext}$ (Items Per Second)')
plt.ylabel(r'Throughput $\lambda_{out}$ (Items Per Second)')
for nts, ds in zip(sim_lengths, ys):
    plt.errorbar(
        arrival_rates, [np.mean(y) for y in ds], yerr=[std_err(y) for y in ds],
        label='Simulated Session Length = %d Reviews' % nts)
plt.axvline(x=theoretical_phase_transition_threshold,
            label=r'Phase Transition Threshold (Theoretical)', linestyle='--')
plt.legend(loc='best')
plt.savefig(os.path.join('figures', 'lqn', 'throughput-vs-arrival-rate-vs-simulated-session-length.pdf'))
plt.show()

# %%

with open(os.path.join('results',
                       'throughput-vs-arrival-rate-vs-simulated-session-length.pkl'),
          'wb') as f:
    pickle.dump((arrival_rates, ys, sim_lengths), f, pickle.HIGHEST_PROTOCOL)

# %% md

# If
# difficulties
# are
# item - specific, does
# creating
# parallel
# queueing
# systems
# help?

# %%

arrival_rates = np.arange(0.001, 0.15, 0.001)
sim_length = 1000
num_systems_set = range(1, 11)

# %%

# from ext_lqn_properties.ipynb
optimal_review_rates = [
    [0.045399259994105136, 0.03847275719791948, 0.03431551304800672,
     0.031248391825640848, 0.02572273483748578],
    [0.01607388427821009, 0.028716279906786, 0.01418113985931187,
     0.02399628349725895, 0.013071220952213317,
     0.021141555920711676, 0.012316601866189176, 0.018942481766948866,
     0.010864112665063222, 0.015106339580169872],
    [0.009574995362327697, 0.013960067171853526, 0.02109463480774494,
     0.00858820264078027, 0.011980694695535053,
     0.017530535748704914, 0.00801450278574112, 0.01080151621076895,
     0.015369501281372995, 0.0076332471983635635,
     0.009958108467235074, 0.013670630601002836, 0.006884757991376857,
     0.008401291880749036, 0.010749861464653288],
    [0.006714438008920156, 0.009237688453210182, 0.011835423783583859,
     0.016773213565409055, 0.006086273015367501,
     0.00803496737482064, 0.010050219385669908, 0.013894393221450899,
     0.005722942578342616, 0.007324062335706905,
     0.008980026869794863, 0.012146727442443267, 0.00548448429102778,
     0.006829139163046215, 0.008194754766998413,
     0.010754894496614424, 0.005011122270095049, 0.005895292937864199,
     0.006774075922036427, 0.008383621517784401],
    [0.005127001550263364, 0.006820580820242886, 0.008371531015656583,
     0.010225672065252102, 0.013979049421944282,
     0.004682866183510203, 0.005989275620204075, 0.007190541306872763,
     0.00863085347246327, 0.011554290753403687,
     0.00442688588752093, 0.005500536428474011, 0.006487332446142775,
     0.0076714396465655905, 0.010081243070342938,
     0.004260252953890548, 0.005165797901650487, 0.005985285503196308,
     0.00695558005354823, 0.00889706475211073,
     0.003927023179050156, 0.004525506061992437, 0.005057213183584693,
     0.005676940018220873, 0.00689257709346182],
    [0.00412572602148644, 0.005362992945624901, 0.006429352327291899,
     0.007564488283251954, 0.009001759456367844,
     0.012017043371052446, 0.003790559014989381, 0.004744148390001485,
     0.005569119841589949, 0.006449655310347438,
     0.0075670530359376416, 0.009916297749941835, 0.003597894068777136,
     0.004381775745773768, 0.005059433818363058,
     0.005782936024405004, 0.006702006503247498, 0.008639533531856102,
     0.0034732157615651534, 0.004136386902729236,
     0.004701815726974527, 0.005298739572206223, 0.006048611381527559,
     0.00760565575491652, 0.003222540185218408,
     0.003662443460112906, 0.00403136487467274, 0.004415692124334982,
     0.004892211612597641, 0.005864239644389311],
    [0.0034401633349379116, 0.004394704157125355, 0.005186881477660381,
     0.0059814445727146195, 0.006876825737061398,
     0.008046795771934807, 0.010559628117936837, 0.0031757313190138073,
     0.003910920837911669, 0.004523219770707268,
     0.005138901320564553, 0.005834131631513151, 0.006744225452490461,
     0.008702401851729078, 0.0030240401605374175,
     0.0036285269860739803, 0.004131519129134704, 0.004637289911571074,
     0.005208717488592617, 0.00595760008874033,
     0.0075733194721484855, 0.002926320252875975, 0.003438916136205053,
     0.0038600836076054467, 0.00427936448089452,
     0.004748680369203616, 0.00535773907644157, 0.006653618536501655,
     0.0027290330866415486, 0.003070008926448823,
     0.003345952442723963, 0.003617426242653257, 0.003917980127278529,
     0.004303562296616209, 0.005110681849743032],
    [0.0029431275158235135, 0.0037082224895140787, 0.0043268822702673135,
     0.004925013161090805, 0.005560113570175186,
     0.006298239392313956, 0.007282382651210947, 0.009431981371507974,
     0.002727649005498385, 0.0033166049799372727,
     0.0037944349700944654, 0.004257502451900565, 0.0047501414270776,
     0.005323661589448345, 0.006089516377738369,
     0.007764881354195466, 0.002604247725947978, 0.0030885985788751606,
     0.0034811663437407433, 0.003861538854976871,
     0.004266312422755438, 0.004737857206584536, 0.005368292619809832,
     0.006751192742917353, 0.002525038731676661,
     0.0029365253223470676, 0.0032661403715068065, 0.0035826151872423457,
     0.003916626337287045, 0.0043025787575814815,
     0.00481401565348028, 0.005921396686000432, 0.002364590827360334,
     0.0026389232162923595, 0.002855592702119905,
     0.0030613834668636623, 0.003276474466994089, 0.003522644695641686,
     0.0038454685865182605, 0.004533893590042749],
    [0.0025672266787736156, 0.0031980909444892804, 0.003698576382497309,
     0.0041704505568082055, 0.004653476113431633,
     0.0051823447605420245, 0.0058091667021883935, 0.006656823034227595,
     0.008532099465993204, 0.0023872752517825166,
     0.002872682111864994, 0.0032590014448035673, 0.0036240565467501274,
     0.003998423747529171, 0.004408973098197347,
     0.004896260255356827, 0.005556106208709167, 0.007017828670240587,
     0.002284364303991858, 0.002683632284553226,
     0.003001055220896595, 0.003300917551748293, 0.003608460289737564,
     0.003945865808164509, 0.004346635047784988,
     0.004889992851317649, 0.0060969521090462955, 0.002218503429820102,
     0.002558225532599843, 0.0028253460230545173,
     0.003075561925905218, 0.0033302671312246787, 0.003607704544024131,
     0.003934832932856721, 0.004374723998180398,
     0.00533993213383286, 0.0020847271648438016, 0.002311633767453025,
     0.0024876942580310995, 0.0026509603375305824,
     0.0028156871399817435, 0.0029936045076368475, 0.0032015865400330874,
     0.0034785813597489243, 0.004077684309311139],
    [0.00227355667675018, 0.002805236836620471, 0.0032209116758492025,
     0.0036057083730374954, 0.003989892160231512,
     0.0043953996973899, 0.004848079897839314, 0.005391966947519379,
     0.00613527079964084, 0.007796363594529018,
     0.002120348223628812, 0.0025292816269610312, 0.002849968885186416,
     0.003147474253595381, 0.0034450261101616874,
     0.0037595685880610036, 0.004111178627856744, 0.004534161398886372,
     0.005112916878636409, 0.006407798093046678,
     0.002032835685961884, 0.0023692581475593714, 0.0026327859726142794,
     0.0028771706280849574, 0.00312158927159844,
     0.003380028387972353, 0.003669066348227619, 0.00401704963475816,
     0.004493780090372081, 0.005563296418237351,
     0.0019769691003886968, 0.002263586712766765, 0.002485770365355199,
     0.002690184421709366, 0.0028932097241258156,
     0.003106484759524572, 0.003343476885060724, 0.0036268807402428196,
     0.0040121676307676175, 0.00486649155814749,
     0.0018632300009208923, 0.002054965563744719, 0.0022017373363445885,
     0.0023354982935077867, 0.002467260542173357,
     0.0026046137205851167, 0.0027560872050753347, 0.0029357950123730468,
     0.0031779163834028913, 0.003707486585455173]]

# %%

num_decks = 5
normalize = lambda x: x / x.sum()
optimal_review_rates = np.array(
    [normalize(np.reshape(x, (y, num_decks))) for x, y in
     zip(optimal_review_rates, num_systems_set)])

# %%

# from ext_lqn_properties.ipynb
transition_thresholds = [0.015048737992973047, 0.015797495306389367,
                         0.01599484746534838, 0.016079634446755746,
                         0.016125055496048188,
                         0.016152717166862837, 0.01617103990361111,
                         0.016183926264181508, 0.01619340146712257,
                         0.016200613478280345]

# %%

num_items = 100
difficulty_of_item = np.random.exponential(1 / difficulty_rate, num_items)

# %%

ys = [[[run_sim(x, num_items, difficulty_rate, difficulty_of_item,
                review_rates, work_rate - x, sim_length) for _ in
        range(num_sim_repeats)] for x in arrival_rates] for y, z in
      zip(num_systems_set, optimal_review_rates)]

# %%

plt.xlabel(r'Arrival Rate $\lambda_{ext}$ (Items Per Second)')
plt.ylabel(r'Throughput $\lambda_{out}$ (Items Per Second)')
for ns, ds, th in zip(num_systems_set, ys, transition_thresholds):
    plt.errorbar(
        arrival_rates, [np.mean(y) for y in ds], yerr=[std_err(y) for y in ds],
        label=r'Simulated ($n = %d$)' % ns)
    plt.axvline(x=th, label=r'Predicted ($n = %d$)' % ns, linestyle='--')
plt.legend(loc='best')
plt.savefig(os.path.join('figures', 'lqn',
                         'throughput-vs-arrival-rate-vs-num-systems.pdf'))
plt.show()

# %%

with open(os.path.join('results',
                       'throughput-vs-arrival-rate-vs-num-systems.pkl'),
          'wb') as f:
    pickle.dump((arrival_rates, ys, num_systems_set, transition_thresholds), f,
                pickle.HIGHEST_PROTOCOL)

# %%

