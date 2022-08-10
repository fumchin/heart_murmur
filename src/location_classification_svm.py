import pandas as pd
import librosa, librosa.feature
import numpy as np
from sklearn.svm import SVC
import os.path, os
from glob import glob
from tqdm import tqdm
from sklearn.svm import SVC
import sklearn.metrics
import config as cfg
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



def preprocess(audio_file_list, saved_path, sr, mel_num, hop_length):
    for count, audio_file in enumerate(tqdm(audio_file_list)):
        y, sr = librosa.load(audio_file, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_num, hop_length=hop_length)
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(S)
        S_scaled = scaler.transform(S)
        file_name = os.path.splitext(os.path.basename(audio_file))[0]
        # save feature
        np.save(os.path.join(saved_path, file_name+'.npy'), S_scaled)



if __name__ == '__main__':
    dataset_name = 'the-circor-digiscope-phonocardiogram-dataset-1.0.3'
    data_path = os.path.join('..', 'dataset', dataset_name, 'training_data')
    audio_file_list = glob(os.path.join(data_path, '*.wav'))
    preprocess_path = os.path.join('..', 'dataset', dataset_name, 'preprocess')
    print(os.path.exists(preprocess_path))
    # audio preprocess
    if not os.path.exists(os.path.join(preprocess_path)):
        os.makedirs(preprocess_path)

        preprocess(audio_file_list = audio_file_list, saved_path = preprocess_path, sr = cfg.sr, mel_num=cfg.mel_num, hop_length=cfg.hop_length)
        feature_file_list = glob(os.path.join(preprocess_path, '*.npy'))
    else:
        feature_file_list = glob(os.path.join(preprocess_path, '*.npy'))
        # print(feature_file_list)
        # feature_file_list = glob(preprocess_path+'\*')


    position_list = []
    # read feature file
    features = None
    frame_length = cfg.frame_length
    for count, feature_file in (enumerate(tqdm(feature_file_list))):
        current_feature = np.load(feature_file)[:, :frame_length]
        if(current_feature.shape[1] < frame_length):
            continue
        basename = os.path.splitext(os.path.basename(feature_file))[0]
        if basename.split('_')[1] == 'Phc':
            continue
        position_list.append(basename.split('_')[1])
        # current_feature = np.mean(current_feature, axis=0)
        current_feature = current_feature.flatten()
        # current_feature = np.reshape(current_feature, (1, current_feature.shape[0], current_feature.shape[1]))
        if (count == 0) | (features is None):
            features = current_feature
        else:
            # len = min(len, current_feature.shape[1])
            features = np.vstack((features, current_feature))
        # if count == 1000:
        #     break
    X_train, X_test, y_train, y_test = train_test_split(features, position_list, test_size=0.2, random_state=0)
    classifier = SVC(verbose=True)
    classifier.fit(X=X_train, y=y_train)
    acc = classifier.score(X_test, y_test)
    print(acc)
    y_pred = classifier.predict(X_test)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)


    # audio file

    # print(position_list)

