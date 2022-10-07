from torch.utils.data import Dataset

import numpy as np
import config_classification as cfg
import os, os.path
from glob import glob
from tqdm import tqdm

import pandas as pd



class PhonocardiogramDataset(Dataset):
    def __init__(self, features_dir, data_dir):
        self.frame_length = cfg.frame_length
        self.features_dir = features_dir
        self.data_dir = data_dir
        # self.features_file_list = None
        self.features_file_list = glob(os.path.join(self.features_dir, '*.npy'))
        self.features_file_list = [file for file in self.features_file_list if 'Phc' not in file]
        # self.features, self.position_list = self._get_features_array_and_labels(features_dir)

    def __len__(self):
        return len(self.features_file_list)

    def __getitem__(self, index):

        current_features_file = self.features_file_list[index]

        features = np.load(current_features_file)[:, :self.frame_length]

        features = features.T
        features = np.reshape(features, (1, features.shape[0], features.shape[1]))


        basename = os.path.splitext(os.path.basename(current_features_file))[0]
        # file_name = basename.split('.')[0]
        patient_id = basename.split('_')[0]
        # df = pd.read_csv(os.path.join(self.data_dir, file_name+'.tsv'))
        with open(os.path.join(self.data_dir, patient_id+'.txt')) as f:
            lines = f.readlines()
            for line in lines:
                if 'Murmur' in line:
                    murmur = line.split(': ')[1].strip()
                    # print(murmur)
                    if murmur == 'Absent':
                        murmur = 0
                    elif murmur == 'Present':
                        murmur = 1
                    else:
                        murmur = 2
                    break
        # print(cfg.murmur_dict[murmur])
        return features, murmur, patient_id


    def _get_features_array_and_labels(self, features_dir):
        features_file_list = glob(os.path.join(features_dir, '*.npy'))
        position_list = []
        features = None
        for count, feature_file in (enumerate(tqdm(features_file_list))):
            current_feature = np.load(feature_file)[:, :self.frame_length]
            current_feature = current_feature.T
            current_feature = np.reshape(current_feature, (1, 1, current_feature.shape[0], current_feature.shape[1]))
            if(current_feature.shape[2] < self.frame_length):
                continue
            basename = os.path.splitext(os.path.basename(feature_file))[0]
            # eliminate the Phc label
            if basename.split('_')[1] == 'Phc':
                continue
            position_list.append(cfg.location_dict[basename.split('_')[1]])
            # concatenate data
            if (count == 0) | (features is None):
                features = current_feature
            else:
                features = np.vstack((features, current_feature))
            if count == 200:
                break

        print(features.shape)
        return features, position_list

if __name__ == "__main__":
    dataset_name = 'the-circor-digiscope-phonocardiogram-dataset-1.0.3'
    features_dir = os.path.join('../dataset', dataset_name, 'preprocess')
    data_dir = os.path.join('../dataset', dataset_name, 'training_data')
    pcd = PhonocardiogramDataset(features_dir=features_dir, data_dir=data_dir)
    signal, label = pcd[100]
    print(signal, label)
