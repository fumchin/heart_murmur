from torch.utils.data import Dataset

import numpy as np
import config as cfg
import os, os.path
from glob import glob
from tqdm import tqdm



class PhonocardiogramDataset(Dataset):
    def __init__(self, features_dir):
        self.frame_length = cfg.frame_length
        self.features_dir = features_dir
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
        position = basename.split('_')[1]
        return features, cfg.location_dict[position]
        # return self.features[index, :, :, :], self.position_list[index]


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
    pcd = PhonocardiogramDataset(features_dir=features_dir)
    signal, label = pcd[3]
    print(signal, label)
