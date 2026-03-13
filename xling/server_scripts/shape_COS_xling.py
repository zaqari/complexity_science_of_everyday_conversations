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
from kgen2.CEDA.calc.deconstructed import processor
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
PATH = '/home/zprosen/d/shapeoflang2/'


############## ITKIN PARAMETERS
RAW_PATH = os.path.join(PATH, 'to_itkin')

CKPTS_PATH = os.path.join(PATH, 'ckpts-itkin')
if not os.path.exists(CKPTS_PATH):
    os.mkdir(CKPTS_PATH)


############### ROSEN PARAMETERS
# RAW_PATH = os.path.join(PATH, 'to_rosen_xling')
#
# CKPTS_PATH = os.path.join(PATH, 'ckpts-rosen')
# if not os.path.exists(CKPTS_PATH):
#     os.mkdir(CKPTS_PATH)


############### Completed files checklist
completed = [None]
if os.path.exists(os.path.join(PATH,'completed.txt')):
    completed = [l.strip() for l in open(os.path.join(PATH,'completed.txt'), 'r').readlines()]
else:
    f = open(os.path.join(PATH,'completed.txt'), 'w')
    f.write('')
    f.close()

level = [7,-1]

print(PATH, CKPTS_PATH, '\n', completed[-1], '\n', level, '\n\n')



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
subids = np.array([f for f in sorted(os.listdir(RAW_PATH)) if (not f.startswith('.')) and (f not in completed)], dtype='object')


#############################################################################
### starting from a saved checkpoint!
#############################################################################
print(subids, '\n\n')


#############################################################################
### Process
#############################################################################
with torch.no_grad():
    for i, submission_id in enumerate(subids):

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

        df = df.loc[
            (~df['x_utterance'].isna())
            & (~df['y_utterance'].isna())
        ]
        # df['conversation_id'] = submission_id

        meta_data_cols = [col for col in list(df) if col not in ['x_utterance', 'y_utterance']]

        print('===]{}[==='.format(submission_id))
        print('{}/{}'.format(i+1, len(subids)))

        print(len(df), 'total contextual units')

        if 'conversation_id' in df.columns:
            print(df['conversation_id'].nunique(), 'conversations')

        if len(df) > 0:
            try:
                # GRAPH = ceda_model(
                #     sigma=sigma,
                #     device='cuda',
                #     wv_model='roberta-base',
                #     wv_layers=level
                # )

                GRAPH = processor(
                    wv_model='xlm-roberta-base',
                    wv_layers=level,
                    device=DEV,
                    checkpoint_location=os.path.join(CKPTS_PATH,submission_id.split('.')[0])
                )

                GRAPH.fit(
                    x=df['x_utterance'].values,
                    y=df['y_utterance'].values,
                )

                outpuf_df = GRAPH.df()
                # outpuf_df = pd.concat([outpuf_df,df[meta_data_cols]],axis=1)

                outpuf_df.to_parquet(
                    os.path.join(
                        CKPTS_PATH,
                        submission_id.split('.')[0],
                        '{}-fin.parquet'.format(submission_id.split('.')[0])
                    )
                )

                del outpuf_df

                # GRAPH.meta_data = df[meta_data_cols].to_dict(orient='records')
                #
                # GRAPH.checkpoint(
                #     os.path.join(
                #         PATH,
                #         str(output_name).format(submission_id)
                #     )
                # )

            except Exception as err:

                torch.cuda.empty_cache()

                err_path = os.path.join(PATH, 'ERRORs.txt')

                if os.path.exists(err_path):
                    f  = open(err_path, 'a')
                    f.write(str(submission_id)+ '\t' + str(err) + '\n')
                    f.close()

                else:
                    f = open(err_path, 'w')
                    f.write(str(submission_id)+ ' ' + str(err) + '\n')
                    f.close()

            f = open(os.path.join(PATH, 'completed.txt'), 'a')
            f.write(submission_id+'\n')
            f.close()


print('=======][=======\n')
