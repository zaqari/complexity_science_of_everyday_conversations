import os
import sys
import shutil
import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm

data_path = '/home/zprosen/d/shapeoflang/'
meta_data_file = os.path.join(data_path, 'all-candor-meta_data.csv')
candor_path = os.path.join(data_path, 'candor/data')

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

for file_id_no, f in enumerate(files):
    print(f)
    temp_file = os.path.join(candor_path, f.split('/')[-1].split('.zip')[0])

    # Unzip the file
    with zipfile.ZipFile(f, 'r') as zf:
        zf.extractall(temp_file)

    # get files to be used
    main_survey_file = [fi for fi in grab_all_files(temp_file, file_type='survey.csv') if
                        (not fi.startswith('__MACOSX/')) and (not 'transcription/' in fi)]

    for fi in tqdm(main_survey_file):
        # create CEDA ready corpus!
        try:

            dfs = pd.read_csv(fi)

            if os.path.exists(meta_data_file):
                dfs.to_csv(meta_data_file, index=False, header=False, encoding='utf-8', mode='a')
            else:
                dfs.to_csv(meta_data_file, index=False, encoding='utf-8')


        except Exception as ex:
            print(ex)

    shutil.rmtree(temp_file)
    print('=======][=======\n\n')