import pandas as pd
import torch
import numpy as np
import os
from datetime import datetime as dt
from tqdm import tqdm
import re
from itertools import product
# If on the remote server
# from kgen2.mutual_information.end_to_end_analysis import analyzer
# from kgen2.mutual_information.fastGraph import fastGraphWithAnalyzer as FGA
# from kgen2.mutual_information.entropy import entropy_cdf
# from kgen2.LM.LM.RoBERTa import RoBERTa
# from shared.CEDA import ceda_model
from kgen2.CEDA.calc.deconstructed import processor
import warnings; warnings.filterwarnings('ignore')

# from kgen2.CEDA.calc.wv_model import wv
# from kgen2.CEDA.calc.fastGraph import justCosinesFastGraph as FGA
# from kgen2.CEDA.calc.entropy import just_cosines
# from kgen2.CEDA.calc.end_to_end_analysis import analyzer

###########################################################################################
# Basic set-up
###########################################################################################
print('CUDA:', torch.cuda.is_available())

DEV = 'cuda'

start = dt.now()
DATA_PATH = '/home/zprosen/d/shapeoflang2/'

RAW_PATH = os.path.join(DATA_PATH, 'to_itkin')

CKPTS_PATH = os.path.join(DATA_PATH, 'ckpts-itkin')

PROCESSED_PATH = os.path.join(CKPTS_PATH, 'lme-ready')
if not os.path.exists(PROCESSED_PATH):
    os.mkdir(PROCESSED_PATH)

output_name = 'graph-obj-{}.pt'

completed = [None]

meta_data_cols = ['conversation_id', 'corpus', 'x_speaker', 'y_speaker', 'x_turn_id', 'y_turn_id']

completed = [l.strip() for l in open(os.path.join(DATA_PATH,'completed-checklist.txt'), 'r').readlines()]

level = [7,-1]

print(RAW_PATH, DATA_PATH, PROCESSED_PATH, '\n', completed[-1], '\n', level, '\n\n')



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

def func(x, lim, sigma=.3):
    x_ = x[:lim] + 1e-9
    p = torch.exp(- (x_**2)/(2*(sigma**2)) )
    return (- torch.log( p )).sum().item()



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




###########################################################################################
### Data set-up
###########################################################################################
subids = np.array(completed, dtype='object')


#############################################################################
### starting from a saved checkpoint!
#############################################################################
print(subids, '\n\n')


#############################################################################
### Process
#############################################################################

with torch.no_grad():
    for i, submission_id in enumerate(subids):
        dir_ = os.path.join(CKPTS_PATH, submission_id.replace('.parquet', ''))

        print('{} | {} | {}/{}'.format(submission_id, dir_, i+1, len(subids)))

        files = [
            os.path.join(dir_, f) for f in os.listdir(dir_)
        ]

        df = []
        for f in tqdm(files):
            df += [pd.read_parquet(f)]
            
            df[-1][['I', 'aCoS']] = 0
            for i in df[-1].index:
                nx = int(df[-1]['nx'].loc[i])
                x = torch.FloatTensor(1-np.array(df[-1]['CoS'].loc[i].replace('[', '').replace(']', '').split(', '), dtype=float))
                df[-1]['aCoS'].loc[i] = x[:nx].mean().item()
                df[-1]['I'].loc[i] = func(x,lim=nx)
            
            del df[-1]['CoS']

        df = pd.concat(df,ignore_index=True)
        print(1-df['aCoS'].mean())
        
        meta_data = pd.read_parquet(
            os.path.join(RAW_PATH, submission_id)
        )

        if 'corpus' not in list(meta_data):
            meta_data['corpus'] = submission_id.replace('.parquet', '')

        meta_data = meta_data[meta_data_cols]

        df = pd.concat([meta_data,df], axis=1)
        
        df.to_parquet(
            os.path.join(PROCESSED_PATH,submission_id),
            engine='fastparquet',
            compression='snappy'
        )

        del meta_data
        del df

        print('====][====')


print('=======]fin[=======\n')
