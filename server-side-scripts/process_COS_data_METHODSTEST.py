###############################################################
#### ISSUES
# (1) How to make sure that conversation id is (a) in corpus and (b) aligned correctly.
# Meta data files have format like this: 'CANDOR-111-null_perm.parquet',
# while data containing files have format like this:
# 'CANDOR-CANDOR-26-c7b2bf7c-ed3b-4adf-82b4-c9232cdffca9'.

# Solution: load meta data files.
###############################################################

import gc

import pandas as pd
import torch
import numpy as np
import os
from datetime import datetime as dt
from tqdm import tqdm
import re
from itertools import product
from convergence_entropy.CEDA.calc._bypass._experimental.deconstructed_cos import processor
import warnings;warnings.filterwarnings('ignore')

###########################################################################################
# Basic set-up
###########################################################################################
print('CUDA:', torch.cuda.is_available())

DEV = 'cuda'

start = dt.now()
DATA_PATH = './d/shapeoflang/'

RAW_PATH = os.path.join(DATA_PATH, 'server_ready')

CKPTS_PATH = os.path.join(DATA_PATH, 'ckpts')

PROCESSED_PATH = os.path.join(DATA_PATH, 'lme-ready')
if not os.path.exists(PROCESSED_PATH):
    os.mkdir(PROCESSED_PATH)

output_name = 'graph-obj-{}.pt'

# completed = [None]

meta_data_cols = ['conversation_id', 'corpus', 'x_speaker', 'y_speaker', 'x_turn_id', 'y_turn_id']

# completed = [l.strip() for l in open(os.path.join(DATA_PATH,'completed-checklist.txt'), 'r').readlines()]

level = [7,-1]

print(RAW_PATH, DATA_PATH, PROCESSED_PATH, '\n', level, '\n\n')



###########################################################################################
# Post-processing-function
###########################################################################################
# I opted in this case to go with the Gaussian Kernel function. It's not hard to
# backtrack necessarily if need be.

# We're also going to focus on surprisal/information. Again, since we're asking about
# information related to the SOURCE, this isn't the right measurement. Entropy is.
# but I'm cognizant enough to know that entropy of the source can be empirically
# found via average information as well (Ch. 2, theorem 4 of AMToC). So we lose very
# little with this change.

# def func(x, lim, sigma=.15):
#     x_ = x[:lim] + 1e-9
#     p = torch.exp(- (x_**2)/(2*(sigma**2)) )
#     return (- torch.log( p )).sum().item()
sigma = .3
N = torch.distributions.HalfNormal(scale=sigma)


def I(x, sigma=sigma):
    x_ = x + 1e-9
    p = np.exp(- (x_**2)/(2*(sigma**2)) )
    return (- np.log( p )).sum().item()

def H(x, sigma=sigma):
    x_ = x + 1e-9
    p = np.exp(- (x_**2)/(2*(sigma**2)) )
    return (-(p * np.log( p ))).sum().item()



###########################################################################################
# Main script
###########################################################################################


###########################################################################################
### Data set-up
###########################################################################################
# subids = np.array(completed, dtype='object')
CORPUS_DOCS = [os.path.join(CKPTS_PATH, f, f+'.parquet') for f in os.listdir(CKPTS_PATH) if (not f.startswith('.'))]
CORPUS_DOCS = [f for f in CORPUS_DOCS if os.path.exists(f)]


#############################################################################
### Process
#############################################################################
with torch.no_grad():

    for ck, CORPUS in enumerate(CORPUS_DOCS):
        print('{} ({}/{})'.format(CORPUS.split('/')[-1], ck+1, len(CORPUS_DOCS)))
        CID = CORPUS.split('/')[-1].replace('.parquet', '')

        outfile = os.path.join(DATA_PATH, 'lme-ready', CID + '.parquet')
        try:
            df = pd.read_parquet(CORPUS)

            df[['aCoS', 'I', 'H']] = 0

            for i in tqdm(df.index):
                nx = int(df['nx'].loc[i])
                x = np.array(str(df['CoS'].loc[i]).replace('[', '').replace(']', '').replace(', ', ',').replace('b', '').replace("'", '').split(','), dtype=float)
                x = 1 - np.where(x > 1, .9999999, x)
                m = x.mean().item()
                x = x[:nx]

                df.loc[i, ['aCoS', 'I', 'H']] = [m, I(x), H(x)]

            df.to_parquet(
                outfile,
                engine='fastparquet',
                compression='snappy'
            )

            del df

            print('====][====')

        except Exception:
            0

print('=======]fin[=======\n')
