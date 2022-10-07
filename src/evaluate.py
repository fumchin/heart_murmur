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
import config_classification as cfg

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess(audio_file_list, sr, mel_num, hop_length):
    feature_list = []
    for count, audio_file in enumerate(tqdm(audio_file_list)):
        y, sr = librosa.load(audio_file, sr=sr)
        min_frame = int(sr * 10)
        hop_frame = int(sr * 5)
        if (y.shape[-1] < min_frame):
            y_list = [librosa.util.fix_length(y, size=min_frame)]
        else:
            y_list = librosa.util.frame(y, frame_length=min_frame, hop_length=hop_frame, axis=0)
        
        for frame_count, current_frame in enumerate(y_list):
            S = librosa.feature.melspectrogram(y=current_frame, sr=sr, n_mels=mel_num, hop_length=hop_length)
            scaler = StandardScaler().fit(S)
            S_scaled = scaler.transform(S)
            file_name = os.path.splitext(os.path.basename(audio_file))[0]
            # save feature
            # np.save(os.path.join(saved_path, file_name+'_'+str(frame_count)+'.npy'), S_scaled)
            feature_list.append(S_scaled)
    return feature_list
        

# def preprocess_patient_data(input_folder):
#     preprocess(input_folder)
#     pass
def weight_accuracy(confusion_matrix):
    cm = confusion_matrix
    present_w = 5
    unknown_w = 3
    absent_w = 1
    weight_acc = (absent_w*cm[0,0] + present_w*cm[1,1] + unknown_w*cm[2,2]) / (absent_w*(cm[0,0]+cm[1,0]+cm[2,0])+present_w*(cm[0,1]+cm[1,1]+cm[2,1])+unknown_w*(cm[0,2]+cm[1,2]+cm[2,2]))
    return weight_acc

if __name__ == '__main__':
    dataset_name = 'the-circor-digiscope-phonocardiogram-dataset-1.0.3'
    training_data_path = os.path.join('..', 'dataset', dataset_name, 'training_data')
    patients_wise_data_path = os.path.join('..', 'dataset', dataset_name, 'evaluation_patient_wise')
    # audio_file_list = glob(os.path.join(data_path, '*.wav'))
    # preprocess_path = os.path.join('..', 'dataset', dataset_name, 'preprocess_10sec')
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    patient_list = os.listdir(patients_wise_data_path)
    # print(patient_list)
    if cfg.model_type == 'CNN':
        model_eval = Cnn_Test().to(device)
    elif cfg.model_type == 'CRNN_fpn':
        model_eval = CRNN_fpn(1, cfg.num_classes, kernel_size=7 * [3], padding=7 * [1], stride=7 * [1], nb_filters=[16,  32,  64,  128,  128, 128, 128],
                attention=True, activation="GLU", dropout=0.5, n_RNN_cell=128, n_layers_RNN=2,
                pooling=[[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]).to(device)
    elif cfg.model_type == 'CRNN':
        model_eval = CRNN(1, cfg.num_classes, kernel_size=7 * [3], padding=7 * [1], stride=7 * [1], nb_filters=[16,  32,  64,  128,  128, 128, 128],
            attention=True, activation="GLU", dropout=0.5, n_RNN_cell=128, n_layers_RNN=2,
            pooling=[[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]).to(device)
    
    checkpoint_path = os.path.join('./stored_data/classification', cfg.model_name)
    best_checkpoint = torch.load(os.path.join(checkpoint_path, 'best_model.pt'))
    model_eval.load_state_dict(best_checkpoint['model_state_dict'])
    model_eval.eval()
    softmax = nn.Softmax(dim=1)
    patient_pred_list = []
    murmur_label_list = []
    
    for patient_count, patient_id in enumerate(patient_list):
        patient_result = None
        # get patient label
        murmur_label = -1
        with open(os.path.join(training_data_path, patient_id+'.txt')) as f:
            lines = f.readlines()
            for line in lines:
                if 'Murmur' in line:
                    murmur = line.split(': ')[1].strip()
                    murmur_label = cfg.murmur_dict[murmur]
                    break
        patient_audio_list = glob(os.path.join(patients_wise_data_path, patient_id,'*.wav'))
        patient_feature_list = preprocess(patient_audio_list, sr = cfg.sr, mel_num=cfg.mel_num, hop_length=cfg.hop_length)
        absent_count = 0
        present_count = 0
        unknown_count = 0
        for feature_count, patient_feature in enumerate(patient_feature_list):
            features = patient_feature[:, :cfg.frame_length]

            features = features.T
            features = np.reshape(features, (1, 1, features.shape[0], features.shape[1]))
            features = torch.from_numpy(features)
            features = features.to(device)
            output = model_eval(features)
            # output = softmax(output)
        #     if feature_count == 0:
        #         patient_result = output
        #     else:
        #         patient_result = torch.add(patient_result, output)
        # # patient_result = softmax(patient_result)
        # patient_pred = torch.argmax(patient_result)
        
            y_pred = torch.argmax(output, dim=1)
            # # # print('dd')
            if y_pred[0] == 1:
                present_count += 1
                # patient_pred = np.array(1)
                
            elif y_pred[0] == 2:
                unknown_count += 1
                # patient_pred = np.array(2)
            else:
                absent_count += 1
                # patient_pred = np.array(0)
                
            if present_count >= int(0.2 * len(patient_feature_list)):
            # if present_count >= 1:
                patient_pred = np.array(1)
                break
            elif unknown_count >= int(0.2 * len(patient_feature_list)):
            # elif unknown_count >= 1:
                patient_pred = np.array(2)
                break
            else:
                patient_pred = np.array(0)

        # if (patient_result[0,2] > patient_result[0,0]):
        #     patient_pred_list.append(np.array([2]))
        # else:
        #     patient_pred_list.append(y_pred.detach().cpu().numpy())
        patient_pred_list.append(patient_pred)
        murmur_label_list.append(murmur_label)
        # if patient_count == 100:
        #     break
        if (patient_count % 5 == 0):
            print(confusion_matrix(murmur_label_list, patient_pred_list))
            print('f1_score: '+ str(f1_score(murmur_label_list, patient_pred_list, average='macro')))
            print('balanced acc: ' + str(balanced_accuracy_score(murmur_label_list, patient_pred_list)))
        # print(y_pred, murmur_label)
        # print('next')
    confusion_matrix = confusion_matrix(patient_pred_list, murmur_label_list)
    print(confusion_matrix)
    print('weight acc: '+ str(weight_accuracy(confusion_matrix)))
    print('f1_score: '+ str(f1_score(murmur_label_list, patient_pred_list, average='macro')))
    print('balanced acc: ' + str(balanced_accuracy_score(murmur_label_list, patient_pred_list)))