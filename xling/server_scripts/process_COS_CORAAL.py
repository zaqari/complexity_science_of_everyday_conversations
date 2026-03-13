###### Documentation/Raison D'Etre
# I'm a moron. In part because I should have split the CORAAL corpus into
# several different corpora. Because I did not I had to more or less set
# up this complicated processing procecure. In the final paper, I won't
# include the script that doesn't split the corpus up, but for now I have
# to manage with this.

import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import gc
import shutil

##### HYPERPARAMS
split_size = 100000

##### FILE PATHS
DATA_PATH = '/home/zprosen/d/shapeoflang2/'

ckpts_path = os.path.join(DATA_PATH, 'ckpts-itkin', 'CORAAL')
meta_data_path = os.path.join(DATA_PATH, 'to_itkin', 'CORAAL.parquet')

tmp_output_data_path = os.path.join(DATA_PATH, '_tmp_coraal')
if not os.path.exists(tmp_output_data_path):
    os.mkdir(tmp_output_data_path)

final_output_path = os.path.join(DATA_PATH, 'ckpts-itkin', 'coraal-lme-ready')
if not os.path.exists(final_output_path):
    os.mkdir(final_output_path)

print(os.path.exists(ckpts_path), os.path.exists(meta_data_path))

##### Function to grab all files in a directory
def grab_all_files(PATH, file_type='.parquet'):
    files = [
        [
            os.path.join(root, f) for f in files
            if f.endswith(file_type) and (not f.startswith('._'))
        ]
        for root, _, files in os.walk(PATH)
    ]
    return sum(files, [])

#### function for processing COS data
# currently using Gaussian kernel
def func(x, lim, sigma=.3):
    x_ = 1-x[:lim]
    p = np.exp(- (x_**2)/(2*(sigma**2)) )
    return (- np.log( p )).sum().item()



#############################################################################
# Splitting up meta data
#############################################################################
meta_data_cols = [
    'corpus', 'conversation_id', 'file',
    'x_turn_id', 'x_speaker',
    'y_turn_id', 'y_speaker'
]

meta_data = pd.read_parquet(meta_data_path)[meta_data_cols]
gc.collect()

steps = int(np.floor(len(meta_data)/split_size))

for i in tqdm(range(steps)):
    meta_data.loc[(i*split_size):((i+1)*split_size)].to_parquet(
        os.path.join(tmp_output_data_path, 'CORAAL-{}.parquet'.format(i)),
        engine='fastparquet',
        compression='snappy'
    )

meta_data.loc[(steps*split_size):].to_parquet(
    os.path.join(tmp_output_data_path, 'CORAAL-{}.parquet'.format(steps)),
    engine='fastparquet',
    compression='snappy'
)

del meta_data
gc.collect()



#############################################################################
# The "show"
#############################################################################

#### Grabbing all the COS containing files
fs = np.array(grab_all_files(ckpts_path), dtype=object)
maximum_val = np.array([v.split('-')[-1].replace('.parquet', '') for v in fs])
maximum_val = np.array([int(v) if (v != 'fin') else -1 for v in maximum_val])
order = maximum_val.argsort()

fs, maximum_val = fs[order][1:], maximum_val[order][1:]

#### Grabbing all the now parsed metadata files
ms = np.array(grab_all_files(tmp_output_data_path), dtype=object)
order = np.array([v.split('-')[-1].replace('.parquet', '') for v in ms])
order = np.array([int(v) if (v != 'fin') else -1 for v in order]).argsort()
ms = ms[order]

#### iteratively processing the data file by file
df = []

for i, f in enumerate(tqdm(fs)):
    df_ = pd.read_parquet(f)
    meta_data = pd.read_parquet(ms[i])
    meta_data.index = range(len(meta_data))

    df_[['aCoS', 'I']] = -404.

    for j in df_.index:

        try:
            v, nx = df_['CoS'].loc[j], df_['nx'].loc[j]

            v = np.array(v.replace('[', '').replace(']','').split(', '),dtype=float)
            av = v.mean()
            I = func(v,nx)

            df_.loc[j, ['aCoS', 'I']] = [av, I]

        except Exception:
            None

    del df_['CoS']

    df_ = pd.merge(
        left=df_, left_index=True,
        right=meta_data, right_index=True,
        how='left'
    )

    df += [df_]

    if ((i+1) % 10) == 0:
        df = pd.concat(df, ignore_index=True)
        df.to_parquet(
            os.path.join(final_output_path, 'CORAAL-{}.parquet'.format(i)),
            engine='fastparquet',
            compression='snappy'
        )

        df = []
        gc.collect()

#### processing the remaining rows.
if len(df):
    df = pd.concat(df, ignore_index=True)
    df.to_parquet(
        os.path.join(final_output_path, 'CORAAL-{}.parquet'.format(len(fs)+1)),
        engine='fastparquet',
        compression='snappy'
    )

#### removing the temporary parsed CORAAL datapath
shutil.rmtree(tmp_output_data_path)