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
from kgen2.CEDA.calc._bypass._experimental.deconstructed_cos import processor

# from kgen2.CEDA.calc.wv_model import wv
# from kgen2.CEDA.calc.fastGraph import justCosinesFastGraph as FGA
# from kgen2.CEDA.calc.entropy import just_cosines
# from kgen2.CEDA.calc.end_to_end_analysis import analyzer

###########################################################################################
# Basic set-up
###########################################################################################
print('CUDA:', torch.cuda.is_available())

# # rosen
# DEV = 'cuda'
# itkin
DEV = 'cuda'

start = dt.now()

PATH = '/home/zprosen/d/shapeoflang3/'

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

            ########## ID if xling file or not
            # if 'xling-' in submission_id:
            #     wv_model_name = 'xlm-roberta-base'
            # else:
            #     wv_model_name = 'roberta-base'
            wv_model_name = 'xlm-roberta-base'

            df = df.loc[
                (~df['x_utterance'].isna())
                & (~df['y_utterance'].isna())
            ]
            # df['conversation_id'] = submission_id

            # meta_data_cols = [col for col in list(df) if col not in ['x_utterance', 'y_utterance']]

            print('===]{}[==='.format(submission_id))
            print('{}/{}'.format(i+1, len(subids)))

            print(len(df), 'total contextual units')

            if 'conversation_id' in df.columns:
                print(df['conversation_id'].nunique(), 'conversations')


            checkpoint_name = os.path.join(CKPTS_PATH, submission_id.split('.')[0])
            if not os.path.exists(checkpoint_name):
                os.mkdir(checkpoint_name)

            outfile = os.path.join(checkpoint_name, checkpoint_name.split('/')[-1] + '.parquet')

            GRAPH = processor(
                wv_model=wv_model_name,
                wv_layers=level,
                device=DEV,
                # checkpoint_location=os.path.join(CKPTS_PATH, checkpoint_name)
            )

            GRAPH.fit(
                x=df['x_utterance'].values,
                y=df['y_utterance'].values,
            )

            gdf = pd.concat([df[meta_data_cols].reset_index(), GRAPH.df()], axis=1)

            gdf.to_parquet(
                outfile,
                engine='fastparquet',
                compression='snappy'
            )

        except Exception:
            torch.cuda.empty_cache()

print('=======][=======\n')
