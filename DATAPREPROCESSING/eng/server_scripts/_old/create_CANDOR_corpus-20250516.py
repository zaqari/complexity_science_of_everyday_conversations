import os
import sys
import shutil
import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm

data_path = '/home/zprosen/d/shapeoflang/'
candor_path = os.path.join(data_path, 'candor/data')
output_file_path = os.path.join(data_path, 'raw')
if not os.path.exists(output_file_path):
    os.mkdir(output_file_path)

def grab_all_files(PATH, file_type='.zip'):
    files = [
        [
            os.path.join(root, f) for f in files
            if f.endswith(file_type) and (not f.startswith('._'))
        ]
        for root, _, files in os.walk(PATH)
    ]
    return sum(files, [])

files = grab_all_files(candor_path)
print(files)

def sort_compose(listy):
    return '-'.join(np.sort(listy))

max_turns = 20
survey_columns = ['race', 'politics', 'age', 'sex', 'edu', 'in_common', 'topic_diversity', 'our_thoughts_synced_up_sr1', 'developed_joint_perspective_sr2', 'shared_thoughts_feels_sr3', 'discussed_real_things_sr4', 'thoughts_became_more_alike_sr5', 'became_certain_of_perception_sr7', 'saw_world_in_same_way_sr8', 'i_think_my_status', 'i_think_your_status']
text_columns = ['turn_id', 'utterance', 'speaker', 'start', 'stop', 'delta']
sort_compose_columns = ['race', 'sex', 'edu']

for file_id_no, f in enumerate(files):
    temp_file = os.path.join(candor_path, f.split('/')[-1].split('.zip')[0])

    # Unzip the file
    with zipfile.ZipFile(f, 'r') as zf:
        zf.extractall(temp_file)

    # get files to be used
    main_data_file = [fi for fi in grab_all_files(temp_file, file_type='_cliffhanger.csv') if
                      not fi.startswith('__MACOSX/')]
    main_survey_file = [fi for fi in grab_all_files(temp_file, file_type='survey.csv') if
                        (not fi.startswith('__MACOSX/')) and (not 'transcription/' in fi)]

    # create CEDA ready corpus!
    try:
        dft = pd.read_csv(main_data_file[0])
        dfs = pd.read_csv(main_survey_file[0])

        ud = dict()
        for uid in dfs['user_id'].unique():
            ud[uid] = dfs[survey_columns].loc[dfs['user_id'].isin([uid])].to_dict(orient='records')[0]

        data = []
        for i in tqdm(dft.index):
            uid = dft['speaker'].loc[i]

            # comparisons to "other"
            next_turns_other = dft.loc[
                                   (dft.index > i)
                                   & (~dft['speaker'].isin([uid]))
                                   ].index.values[:max_turns].tolist()

            # comparisons to "self"
            next_turns_self = dft.loc[
                                  (dft.index > i)
                                  & dft['speaker'].isin([uid])
                                  ].index.values[:max_turns].tolist()

            d_ = {'x_' + col: dft[col].loc[i] for col in text_columns}
            for k, v in ud[uid].items():
                d_['x_' + k] = v

            for t in next_turns_other:
                _uid = dft['speaker'].loc[t]
                d_['self'] = False

                for col in text_columns:
                    d_['y_' + col] = dft[col].loc[t]

                for k, v in ud[_uid].items():
                    d_['y_' + k] = v

                data += [d_.copy()]

            for t in next_turns_self:
                _uid = dft['speaker'].loc[t]
                d_['self'] = True

                for col in text_columns:
                    d_['y_' + col] = dft[col].loc[t]

                for k, v in ud[_uid].items():
                    d_['y_' + k] = v

                data += [d_.copy()]

        if data:
            data = pd.DataFrame(data)
            for col in sort_compose_columns:
                data['combined_' + col] = [sort_compose(data[['x_' + col, 'y_' + col]].loc[idx].values) for idx in
                                           data.index]
            data.to_csv(
                os.path.join(candor_path, str(file_id_no) + '.csv'),
                index=False,
                encoding='utf-8'
            )

    except Exception as ex:
        print(ex)

    shutil.rmtree(temp_file)
    print('=======][=======\n\n')