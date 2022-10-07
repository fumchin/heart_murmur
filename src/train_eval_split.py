from model.CNN_test import Cnn_Test
from model.CRNN_FPN import CRNN_fpn
from model.CRNN import CRNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader_classification import PhonocardiogramDataset

import pandas as pd
import librosa, librosa.feature, librosa.util
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib

import os.path, os
from glob import glob
from tqdm import tqdm
import shutil
import config_classification as cfg

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split

def list_splitter(list_to_split, ratio):
    elements = len(list_to_split)
    middle = int(elements * ratio)
    return [list_to_split[:middle], list_to_split[middle:]]

if __name__ == '__main__':    
    dataset_name = 'the-circor-digiscope-phonocardiogram-dataset-1.0.3'
    training_data_path = os.path.join('..', 'dataset', dataset_name, 'training_data')
    patients_wise_data_path = os.path.join('..', 'dataset', dataset_name, 'patient_wise')
    preprocess_path = os.path.join('..', 'dataset', dataset_name, 'preprocess_10sec')
    evaluation_path = os.path.join('..', 'dataset', dataset_name, 'evaluation_patient_wise')
    patient_list = os.listdir(patients_wise_data_path)
    # train_patient_list, eval_patient_list = list_splitter(patient_list, 0.8)
    absent_count = 0
    present_count = 0
    unknown_count = 0
    
    absent_limit = int(695 * 0.1)
    present_limit = int(179 * 0.1)
    unknown_limit = int(68 * 0.1)
    
    training_preprocess_path = os.path.join('..', 'dataset', dataset_name, 'training_preprocess')
    if not os.path.exists(training_preprocess_path):
        os.makedirs(training_preprocess_path)
    
    for patient_id in tqdm(patient_list):
        # read file first
        with open(os.path.join(training_data_path, patient_id+'.txt')) as f:
            lines = f.readlines()
            for line in lines:
                if 'Murmur' in line:
                    murmur = line.split(': ')[1].strip()
                    murmur_label = cfg.murmur_dict[murmur]
                    break 
        # eval part
        if (murmur == 'Absent') and (absent_count < absent_limit):
            absent_count += 1
            if not os.path.exists(os.path.join(evaluation_path, patient_id)):
                os.makedirs(os.path.join(evaluation_path, patient_id))
                shutil.copytree(os.path.join(patients_wise_data_path, patient_id), os.path.join(evaluation_path, patient_id), dirs_exist_ok=True)
        elif (murmur == 'Present') and (present_count < present_limit):
            present_count += 1
            if not os.path.exists(os.path.join(evaluation_path, patient_id)):
                os.makedirs(os.path.join(evaluation_path, patient_id))
                shutil.copytree(os.path.join(patients_wise_data_path, patient_id), os.path.join(evaluation_path, patient_id), dirs_exist_ok=True)
        elif (murmur == 'Unknown') and (unknown_count < unknown_limit):
            unknown_count += 1
            if not os.path.exists(os.path.join(evaluation_path, patient_id)):
                os.makedirs(os.path.join(evaluation_path, patient_id))
                shutil.copytree(os.path.join(patients_wise_data_path, patient_id), os.path.join(evaluation_path, patient_id), dirs_exist_ok=True)
        # train part
        else:
            train_patient_audio_list = glob(os.path.join(preprocess_path, patient_id+'*.npy'))
            for train_patient_audio in train_patient_audio_list:
                shutil.copy(train_patient_audio, training_preprocess_path)
    # print(len(train_patient_list))
    # print(len(eval_patient_list))
    # for training data split
    
    
    # for train_patient in tqdm(train_patient_list):
    #     train_patient_audio_list = glob(os.path.join(preprocess_path, train_patient+'*.npy'))
    #     for train_patient_audio in train_patient_audio_list:
    #         shutil.copy(train_patient_audio, training_preprocess_path)
            
    # evaluation_path = os.path.join('..', 'dataset', dataset_name, 'evaluation_patient_wise')
    # if not os.path.exists(evaluation_path):
    #     os.makedirs(evaluation_path)
    
    # for eval_patient in tqdm(eval_patient_list):
    #     if not os.path.exists(os.path.join(evaluation_path, eval_patient)):
    #         os.makedirs(os.path.join(evaluation_path, eval_patient))
    #     shutil.copytree(os.path.join(patients_wise_data_path, eval_patient), os.path.join(evaluation_path, eval_patient), dirs_exist_ok=True)