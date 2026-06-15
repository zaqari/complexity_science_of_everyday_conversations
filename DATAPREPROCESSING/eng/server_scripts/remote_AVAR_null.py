import gc
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

DATA_PATH = '/home/zprosen/d/shapeoflang3/'
LME_READY_DIR = os.path.join(DATA_PATH, 'lme-ready')
CHECKLIST_FILE = os.path.join(DATA_PATH, 'completed.txt')

turn_col = 'x_turn_id'
conversation_id_col = 'conversation_id'
# ordinal_col = 'sample_delta'
val_col = 'I'

merge_cols = [conversation_id_col, 'self', turn_col]
# merge_cols = ['self', turn_col]

# f = open(os.path.join(DATA_PATH,CHECKLIST_FILE), 'r')
# subids = np.array([fi.strip() for fi in f.readlines() if fi.strip()], dtype=object)
subids = [f for f in os.listdir(LME_READY_DIR) if (not f.startswith('.')) and f.endswith('.parquet')]



####################################################################
# AVAR fn
####################################################################
def add_avar_col_v1(df, val_col, merge_cols):
    # differentiate turn id cols
    # df[turn_col] = df[conversation_id_col].astype(str) + '-' + df[turn_col].astype(str)

    # calculate period length
    tau = df[merge_cols].value_counts().reset_index()
    tau = tau.rename(columns={'count': 'tau'})

    # add period length col
    df = pd.merge(
        left=df, left_on=merge_cols,
        right=tau, right_on=merge_cols,
        how='left'
    )

    # delete tau df and collect memory
    del tau
    gc.collect()

    # add next step in series
    df[val_col + '_2'] = None
    df[val_col + '_2'].loc[:len(df) - 2] = df[val_col].values[1:]

    # add next, next step in series
    df[val_col + '_3'] = None
    df[val_col + '_3'].loc[:len(df) - 3] = df[val_col].values[2:]

    # calculate AVAR
    df['AVAR'] = (1 / (2 * (df['tau'] ** 2))) * (df[val_col + '_3'] - (2 * df[val_col + '_2']) + df[val_col]) ** 2

    # render comparisons to values that are otherwise erroneous null
    sel = []
    for col in merge_cols:
        v = df[col].values

        # first pass
        sel = v[:-1] != v[1:]
        sel = np.concat([sel, [True]])
        df['AVAR'].loc[sel] = None

        # second pass
        sel = v[:-2] != v[2:]
        sel = np.concat([sel, [True, True]])
        df['AVAR'].loc[sel] = None

        # collect excess memory
        del v
        del sel
        gc.collect()

    sel = np.concat(sel, axis=1).sum(axis=-1) != len(merge_cols)
    sel = np.concat([sel, [False] * 2])

    df['AVAR'].loc[sel] = None

    # collect memory
    # del sel
    del df[val_col + '_2']
    del df[val_col + '_3']
    gc.collect()

    return df

def add_avar_col(df, val_col, merge_cols):
    # differentiate turn id cols
    # df[turn_col] = df[conversation_id_col].astype(str) + '-' + df[turn_col].astype(str)

    # # calculate period length
    # tau = df[merge_cols].value_counts().reset_index()
    # tau = tau.rename(columns={'count': 'tau'})
    #
    # # add period length col
    # df = pd.merge(
    #     left=df, left_on=merge_cols,
    #     right=tau, right_on=merge_cols,
    #     how='left'
    # )
    #
    # # delete tau df and collect memory
    # del tau
    # gc.collect()

    # add next step in series
    df[val_col + '_2'] = None
    df[val_col + '_3'] = None

    groups = df.groupby(by=merge_cols)
    completed = []
    for _, dfi in groups:
        idx = dfi.index
        dfi[val_col + '_2'].loc[idx[:-1]] = dfi[val_col].values[1:]
        dfi[val_col + '_3'].loc[idx[:-2]] = dfi[val_col].values[2:]
        dfi['AVAR'] = (1 / (2 * (len(dfi) ** 2))) * (dfi[val_col + '_3'] - (2 * dfi[val_col + '_2']) + dfi[val_col]) ** 2
        completed+=[dfi]
    df = pd.concat(completed, ignore_index=True)

    # df[val_col + '_2'].loc[:len(df) - 2] = df[val_col].values[1:]
    #
    # # add next, next step in series
    # df[val_col + '_3'] = None
    # df[val_col + '_3'].loc[:len(df) - 3] = df[val_col].values[2:]
    #
    # # calculate AVAR
    # df['AVAR'] = (1 / (2 * (df['tau'] ** 2))) * (df[val_col + '_3'] - (2 * df[val_col + '_2']) + df[val_col]) ** 2

    # render comparisons to values that are otherwise erroneous null
    # sel = []
    # for col in merge_cols:
    #     v = df[col].values
    #     sel += [((v[:-1] == v[1:])[:-1] & (v[:-2] == v[2:])).astype(int).reshape(-1,1)]
    #
    #     # # collect excess memory
    #     # del v
    #     # gc.collect()
    #
    # sel = np.concat(sel, axis=1).sum(axis=-1) != len(merge_cols)
    # sel = np.concat([sel, [False] * 2])
    #
    # df['AVAR'].loc[sel] = None

    # collect memory
    # del sel
    del df[val_col + '_2']
    del df[val_col + '_3']
    gc.collect()

    return df


####################################################################
# The show
####################################################################
for f in tqdm(subids):
    f_ = os.path.join(LME_READY_DIR, f)
    df = pd.read_parquet(f_, engine='fastparquet')
    df['self'] = df['x_speaker'] == df['y_speaker']
    df = add_avar_col(df, val_col=val_col, merge_cols=merge_cols)

    # if doing regression outright, do:
    #

    # if saving AVAR for later, do:
    df.to_parquet(
        f_,
        engine='fastparquet',
        compression='snappy'
    )