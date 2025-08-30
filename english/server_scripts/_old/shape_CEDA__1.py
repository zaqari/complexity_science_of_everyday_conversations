import pandas as pd
import torch
import numpy as np
import os
from datetime import datetime as dt
import re
from itertools import product
# If on the remote server
# from kgen2.mutual_information.end_to_end_analysis import analyzer
# from kgen2.mutual_information.fastGraph import fastGraphWithAnalyzer as FGA
# from kgen2.mutual_information.entropy import entropy_cdf
# from kgen2.LM.LM.RoBERTa import RoBERTa
# from shared.CEDA import ceda_model
from kgen2.CEDA import ceda_model



###########################################################################################
# Basic set-up
###########################################################################################
print('CUDA:', torch.cuda.is_available())

start = dt.now()
PATH = '/home/zprosen/d/shapeoflang/'

RAW_PATH = os.path.join(PATH, 'raw')

CKPTS_PATH = os.path.join(PATH, 'ckpts')
if not os.path.exists(CKPTS_PATH):
    os.mkdir(CKPTS_PATH)

output_name = 'ckpts/graph-obj-{}.pt'

start_at = None
if os.path.exists(os.path.join(PATH,'start_at.txt')):
    start_at = open(os.path.join(PATH,'start_at.txt'), 'r').read()

level = [7,-1]

print(PATH, '\n', start_at, '\n', level, '\n\n')



###########################################################################################
# Graph object definition
###########################################################################################
# GRAPH = ceda_model(
#     sigma=1.5,
#     device='cuda',
#     wv_model='roberta-base',
#     wv_layers=level
# )



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
subids = np.array([f for f in sorted(os.listdir(RAW_PATH)) if not f.startswith('._')], dtype='object')


#############################################################################
### starting from a saved checkpoint!
#############################################################################
if start_at and (start_at in subids):
    subids = subids[(subids==start_at).nonzero()[0][0]:]

print(subids, '\n\n')


#############################################################################
### Process
#############################################################################
for i, submission_id in enumerate(subids):

    df = pd.read_csv(
        os.path.join(RAW_PATH, submission_id)
    )

    df = df.loc[
        (~df['x_utterance'].isna())
        & (~df['y_utterance'].isna())
    ]
    df['conversation_id'] = 'candor-' + submission_id

    meta_data_cols = [col for col in list(df) if col not in ['x_utterance', 'y_utterance']]

    print('===]{}[==='.format(submission_id))
    print('{}/{}'.format(i+1, len(subids)))

    print(len(df), 'total contextual units')

    if len(df) > 0:
        GRAPH = ceda_model(
            sigma=1.5,
            device='cuda',
            wv_model='roberta-base',
            wv_layers=level
        )

        GRAPH.fit(
            x=df['x_utterance'].values,
            y=df['y_utterance'].values,
        )

        # de-facto adding metadata!
        GRAPH.meta_data = df[meta_data_cols].to_dict(orient='records')

        GRAPH.checkpoint(
            os.path.join(
                PATH,
                str(output_name).format(submission_id)
            )
        )

        f = open(os.path.join(PATH, 'start_at.txt'), 'w')
        f.write(submission_id)
        f.close()


print('=======][=======\n')
