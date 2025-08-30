import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf

DATA_LOCATION = '/home/zprosen/d/allan_variance/'

DATA_PATH = os.path.join(DATA_LOCATION, 'allan_var-multiling.parquet')

output_path = os.path.join(DATA_LOCATION, 'av-params-multiling.csv')

print(DATA_PATH, '\n', output_path)
df = pd.read_parquet(DATA_PATH)

df['log_af'] = np.log(df['allan_var'])
df['log_turn'] = np.log(df['sample_delta'])

df = df.loc[
    (~df['log_af'].isna())
    & (df['allan_var'].abs() != np.inf)
]


##########################################
## Main model
##########################################
# model = "log_af ~ log_turn + self"
model = "log_af ~ log_turn"
# model = "log_turn ~ log_af"
##########################################
print(model)

df = df.groupby(by=['groups', 'self'])

results = []

ct = 0
for dfi in tqdm(df):
    ct += 1

    if len(dfi[1]) > 2:
        md = smf.ols(model, dfi[1]).fit()

        results += [
            {
                'x_turn_id': dfi[0][0],
                'null': 'null-' in dfi[0][0],
                'self': dfi[0][1],
                'conversation_id': dfi[1]['conversation_id'].unique()[0],
                'speaker': dfi[1]['x_speaker'].unique()[0],
                'slope': md.params['log_turn'],
                'b0': md.params['Intercept']
            }
        ]

# vals = df[['groups', 'self']].value_counts(sort=False)
# vals = vals[vals > 2].reset_index()
# print(len(vals))
# vals.head()

# results = []
#
# for x_speaker in tqdm(df['groups'].unique()):
#
#     sub = df.loc[
#         df['groups'].isin([x_speaker])
#     ]
#
#     # md = smf.ols(model, sub).fit()
#     #
#     # results += [
#     #     {
#     #         'x_turn_id': x_speaker,
#     #         'x_speaker': sub['x_speaker'].unique()[0],
#     #         'self': md.params['self'],
#     #         'conversation_id': sub['conversation_id'].unique()[0],
#     #         'slope': md.params['log_turn']
#     #     }
#     # ]
#
#     if sub['self'].sum():
#
#         md = smf.ols(model, sub.loc[sub['self']]).fit()
#
#         results += [
#             {
#                 'x_turn_id': x_speaker,
#                 'x_speaker': sub['x_speaker'].unique()[0],
#                 'self': True,
#                 'conversation_id': sub['conversation_id'].unique()[0],
#                 'slope': md.params['log_turn']
#             }
#         ]
#
#     if (~sub['self']).sum():
#
#         md = smf.ols(model, sub.loc[~sub['self']]).fit()
#
#         results += [
#             {
#                 'x_turn_id': x_speaker,
#                 'x_speaker': sub['x_speaker'].unique()[0],
#                 'self': False,
#                 'conversation_id': sub['conversation_id'].unique()[0],
#                 'slope': md.params['log_turn']
#             }
#         ]


results = pd.DataFrame(results)

results.to_csv(
    output_path,
    index=False,
    encoding='utf-8'
)