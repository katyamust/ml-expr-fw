
# this is sample code to create test and train dataframes using stanford original dataset
# sample is taken from here: https://towardsdatascience.com/building-a-sentiment-classifier-using-scikit-learn-54c8e7c5d2f0
# the code is provided to play with nlp_sample and not part of framework

import pandas as pd
import re
from os import system, listdir
from os.path import isfile, join
from random import shuffle


def create_data_frame(folder: str) -> pd.DataFrame:
    '''
    folder - the root folder of train or test dataset
    Returns: a DataFrame with the combined data from the input folder
    '''
    pos_folder = f'{folder}/pos'  # positive reviews
    neg_folder = f'{folder}/neg'  # negative reviews

    def get_files(fld: str) -> list:
        '''
        fld - positive or negative reviews folder
        Returns: a list with all files in input folder
        '''
        return [join(fld, f) for f in listdir(fld) if isfile(join(fld, f))]

    def append_files_data(data_list: list, files: list, label: int) -> None:
        '''
        Appends to 'data_list' tuples of form (file content, label)
        for each file in 'files' input list
        '''
        for file_path in files:
            with open(file_path, 'r') as f:
                text = f.read()
                data_list.append((text, label))
                print(".")

    pos_files = get_files(pos_folder)
    neg_files = get_files(neg_folder)

    data_list = []
    append_files_data(data_list, pos_files, 1)
    append_files_data(data_list, neg_files, 0)
    print("shuffle")
    shuffle(data_list)

    text, label = tuple(zip(*data_list))
    # replacing line breaks with spaces
    text = list(map(lambda txt: re.sub('(<br\s*/?>)+', ' ', txt), text))

    return pd.DataFrame({'text': text, 'label': label})

def main():
    print("train df")
    imdb_train = create_data_frame('aclImdb/train')
    print("test df")
    imdb_test  = create_data_frame('aclImdb/test')

    system("mkdir 'csv'")
    imdb_train.to_csv('csv/imdb_train.csv', index=False)
    imdb_test.to_csv('csv/imdb_test.csv', index=False)

    # imdb_train = pd.read_csv('csv/imdb_train.csv')
    # imdb_test = pd.read_csv('csv/imdb_test.csv')

if __name__ == '__main__':
    main()
