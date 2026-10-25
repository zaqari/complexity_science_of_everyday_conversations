import pandas as pd
import torch
import numpy as np
import os
from datetime import datetime as dt
import re
from itertools import product
from convergence_entropy.CEDA.calc._bypass._experimental.deconstructed_cos import processor

###########################################################################################
# Basic set-up
###########################################################################################
print('CUDA:', torch.cuda.is_available())

# # rosen
# DEV = 'cuda'
# itkin
DEV = 'cuda'

start = dt.now()

PATH = './d/shapeoflang/'

RAW_PATH = os.path.join(PATH, 'server_ready')

CKPTS_PATH = os.path.join(PATH, 'ckpts')
if not os.path.exists(CKPTS_PATH):
    os.mkdir(CKPTS_PATH)

completed = os.listdir(CKPTS_PATH)

level = [7,-1]

print(PATH,  '\n', level, '\n\n')

###########################################################################################
# Main script
###########################################################################################

###########################################################################################
### Data set-up
###########################################################################################
subids = np.array([
    f for f in sorted(os.listdir(RAW_PATH))
    if (not f.startswith('.'))
       and (not os.path.isdir(os.path.join(RAW_PATH, f)))
       and (f.split('.parquet')[0] not in completed)
], dtype='object')

del completed

meta_data_cols = [
    'x_speaker', 'y_speaker',
    'x_turn_id', 'y_turn_id',
    'conversation_id',
    'corpus'
]

# language model to be used
wv_model_name = 'xlm-roberta-base'


#############################################################################
### starting from a saved checkpoint!
#############################################################################
print(subids, '\n\n')


#############################################################################
### Process
#############################################################################
with torch.no_grad():
    for i, submission_id in enumerate(subids):
        try:
            ############# import file
            if submission_id.endswith('.csv'):
                df = pd.read_csv(
                    os.path.join(RAW_PATH, submission_id)
                )

            elif submission_id.endswith('.parquet'):
                df = pd.read_parquet(
                    os.path.join(RAW_PATH, submission_id)
                )

            else:
                df = pd.read_table(
                    os.path.join(RAW_PATH, submission_id),
                    sep='\t'
                )

            # remove bad values
            df = df.loc[
                (~df['x_utterance'].isna())
                & (~df['y_utterance'].isna())
            ]

            print('===]{}[==='.format(submission_id))
            print('{}/{}'.format(i+1, len(subids)))

            print(len(df), 'total contextual units')

            if 'conversation_id' in df.columns:
                print(df['conversation_id'].nunique(), 'conversations')


            checkpoint_name = os.path.join(CKPTS_PATH, submission_id.split('.')[0])
            if not os.path.exists(checkpoint_name):
                os.mkdir(checkpoint_name)

            outfile = os.path.join(checkpoint_name, checkpoint_name.split('/')[-1] + '.parquet')

            # reset and create new processing object.
            GRAPH = processor(
                wv_model=wv_model_name,
                wv_layers=level,
                device=DEV,
                # checkpoint_location=os.path.join(CKPTS_PATH, checkpoint_name)
            )

            # creates a string per comparison of x to y utterance containining:
            #   (1) max COS for each token in x, concatenated with
            #   (2) max COS for each token in y
            #   (3) columns contianing the number of tokens in the utterance x (nx),
            #       and the number of tokens in the utterance y (ny)
            GRAPH.fit(
                x=df['x_utterance'].values,
                y=df['y_utterance'].values,
            )

            # concat and save
            gdf = pd.concat([df[meta_data_cols].reset_index(), GRAPH.df()], axis=1)

            gdf.to_parquet(
                outfile,
                engine='fastparquet',
                compression='snappy'
            )

        except Exception:
            torch.cuda.empty_cache()

print('=======][=======\n')
