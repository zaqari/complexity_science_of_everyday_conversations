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
from kgen2.CEDA.calc._bypass._experimental.deconstructed_cos import processor
import warnings;warnings.filterwarnings('ignore')

###########################################################################################
# Basic set-up
###########################################################################################
print('CUDA:', torch.cuda.is_available())

DEV = 'cuda'

start = dt.now()
DATA_PATH = '/home/zprosen/d/shapeoflang3/'

RAW_PATH = os.path.join(DATA_PATH, 'null')

CKPTS_PATH = os.path.join(DATA_PATH, 'null-ckpts')

PROCESSED_PATH = os.path.join(DATA_PATH, 'null-lme-ready')
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
# backtrack necessarily if need be, and while I am admittedly pretty annoyed with
# Gus for not focusing on what I was bringing him in on and instead trying to
# dictate analytical strategy, I'll throw him this one bone and call it done (beyond
# this he's out or just needs to handle his own lack of IT/calculus-based stats
# background. Literally not my problem.).

# We're also going ot focus on surprisal/information. Again, since we're asking about
# information related to the source, this isn't the right measurement. Entropy is.
# but I'm cognizant enough to know that entropy of the source can be empirically
# found via average information as well (Ch. 2, theorem 4 of AMToC).

# def func(x, lim, sigma=.15):
#     x_ = x[:lim] + 1e-9
#     p = torch.exp(- (x_**2)/(2*(sigma**2)) )
#     return (- torch.log( p )).sum().item()
sigma = .3
N = torch.distributions.HalfNormal(scale=sigma)


def I1(x, sigma=sigma):
    x_ = x + 1e-9
    p = np.exp(- (x_**2)/(2*(sigma**2)) )
    return (- np.log( p )).sum().item()

def I2(x, sigma=sigma):
    x_ = x + 1e-9
    p = np.exp(- (x_**2)/(2*(sigma**2)) )
    CONST = 1/(sigma*np.sqrt(2*np.pi))
    p = CONST * p
    return (- np.log( p )).sum().item()

def H1(x, sigma=sigma):
    x_ = x + 1e-9
    p = np.exp(- (x_**2)/(2*(sigma**2)) )
    return (-(p * np.log( p ))).sum().item()

def H2(x, sigma=sigma):
    x_ = x + 1e-9
    p = np.exp(- (x_**2)/(2*(sigma**2)) )
    CONST = 1 / (sigma * np.sqrt(2 * np.pi))
    p = CONST * p
    return (-(p * np.log( p ))).sum().item()

def H3(x):
    p = 1-N.cdf(torch.FloatTensor(x))
    return - (p * torch.log(p)).sum().item()

###########################################################################################
# Emergency big data cruncher
###########################################################################################
# not finished . . . going to go with instead a pairwise comparison between comments.
# def emergency(x,y,GRAPH, split_at: int=500):
#     GRAPH.GRAPH.M = torch.cat([GRAPH.GRAPH.M, torch.zeros(size=(len(y),2))], dim=0)
#     GRAPH.GRAPH.N = torch.cat([GRAPH.GRAPH.M, torch.zeros(size=(len(y),2))], dim=0)
#
#     spans = int(np.floor(len(x) / split_at))
#
#     steps = [(i * split_at, (i + 1) * split_at) for i in range(spans)]
#     if steps[-1][-1] < len(ex):
#         steps += [(steps[-1][-1], None)]
#
#     # split text and feed values to M and N



###########################################################################################
# Main script
###########################################################################################
# def get_conversation_id_from_dir(x):
#     if x.startswith('CANDOR'):
#         return '-'.join(x.split('-')[3:])
#     elif x.startswith('xling-'):
#         if ('croation-cro' in x):
#             None
#     else:
#         return '-'.join(x.split('-')[2:])


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
                # print(df['CoS'].loc[i])
                x = np.array(str(df['CoS'].loc[i]).replace('[', '').replace(']', '').replace(', ', ',').replace('b', '').replace("'", '').split(','), dtype=float)
                # x = np.array(df['CoS'].loc[i], dtype=float)
                x = 1 - np.where(x > .9999999, .9999999, x)
                m = x.mean().item()
                x = x[:nx]
                # x = torch.FloatTensor(x)

                df.loc[i, ['aCoS', 'I', 'H']] = [m, I1(x), H3(x)]

            # del df['CoS']

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
