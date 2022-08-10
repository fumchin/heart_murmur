from model.CNN import Cnn_Test
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import PhonocardiogramDataset

import pandas as pd
import librosa, librosa.feature
import numpy as np
import math
import matplotlib.pyplot as plt

import os.path, os
from glob import glob
from tqdm import tqdm
import config as cfg

from sklearn.svm import SVC
from sklearn.svm import SVC
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


LOSS_MIN = math.inf

def preprocess(audio_file_list, saved_path, sr, mel_num, hop_length):
    for count, audio_file in enumerate(tqdm(audio_file_list)):
        y, sr = librosa.load(audio_file, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_num, hop_length=hop_length)
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(S)
        S_scaled = scaler.transform(S)
        file_name = os.path.splitext(os.path.basename(audio_file))[0]
        # save feature
        np.save(os.path.join(saved_path, file_name+'.npy'), S_scaled)


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    return train_dataloader
    
def train(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device, epochs, checkpoint_path):
    global LOSS_MIN
    training_loss_list = []
    validation_loss_list = []
    epoch_list = []
    for i in range(epochs):
        print(f"Epoch {i+1}")
        epoch_list.append(i)
        training_loss, validation_loss = train_single_epoch(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device)

        training_loss_list.append(training_loss)
        
        validation_loss_list.append(validation_loss)
        
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_path, f'checkpoint_epoch_{i}'))
        
        if training_loss < LOSS_MIN:
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_path, f'best_model'))
            
            LOSS_MIN = training_loss
            
            output_txt = os.path.join(checkpoint_path, 'best.txt')
            with open(output_txt, 'w') as f:
                f.write(f'training_loss: {training_loss}\n')
                f.write(f'epoch: {i}\n')
        
        print("---------------------------")
        if (i+1) % 5 == 0:
            # draw training & validation loss
            plt.plot(epoch_list, training_loss_list, 'b')
            plt.plot(epoch_list, validation_loss_list, 'r')
            plt.savefig(f'{cfg.model_name}_loss.png')
    
    print("Finished training")
    plt.plot(epoch_list, training_loss_list, 'b')
    plt.plot(epoch_list, validation_loss_list, 'r')
    plt.savefig(f'{cfg.model_name}_loss.png')
    
    
    
    

def train_single_epoch(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device):
    model.train()
    training_loss = 0
    for input, target in train_dataloader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)
        training_loss += loss.item()
        # backpropagate error and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    training_loss = training_loss / len(train_dataloader)
    print(f"loss: {training_loss}")
    
    model.eval()
    validation_loss = 0
    for input, target in validation_dataloader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        val_loss = loss_fn(prediction, target)
        validation_loss += val_loss.item()
    validation_loss = validation_loss / len(validation_dataloader)
    print(f"val_loss: {validation_loss}")
    
    return training_loss, validation_loss

    
    

    
if __name__ == '__main__':
    dataset_name = 'the-circor-digiscope-phonocardiogram-dataset-1.0.3'
    data_path = os.path.join('..', 'dataset', dataset_name, 'training_data')
    audio_file_list = glob(os.path.join(data_path, '*.wav'))
    preprocess_path = os.path.join('..', 'dataset', dataset_name, 'preprocess')
    
   
    
    # audio preprocess
    if not os.path.exists(os.path.join(preprocess_path)):
        os.makedirs(preprocess_path)

        preprocess(audio_file_list = audio_file_list, saved_path = preprocess_path, sr = cfg.sr, mel_num=cfg.mel_num, hop_length=cfg.hop_length)
        feature_file_list = glob(os.path.join(preprocess_path, '*.npy'))
    else:
        feature_file_list = glob(os.path.join(preprocess_path, '*.npy'))

    
    checkpoint_path = os.path.join('./stored_data', cfg.model_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    dataset = PhonocardiogramDataset(preprocess_path)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    torch.manual_seed(0)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # validation_size = int(len(test_dataset) * 0.5)
    # test_size = len(test_dataset) - validation_size
    # test_dataset, validation_dataset = torch.utils.data.random_split(test_dataset, [test_size, validation_size])
    
    train_dataloader = create_data_loader(train_dataset, cfg.batch_size)
    # validation_dataloader = create_data_loader(validation_dataset, cfg.batch_size)
    test_dataloader = create_data_loader(test_dataset, cfg.batch_size)

    
    # training stategies
    model = Cnn_Test().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    model.train()
    train(model=model, train_dataloader=train_dataloader, validation_dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device, epochs=cfg.epochs, checkpoint_path=checkpoint_path)
    # print('fff')
    # X_train, X_test, y_train, y_test = train_test_split(features, position_list, test_size=0.2, random_state=0)
    # classifier = SVC(verbose=True)
    # classifier.fit(X=X_train, y=y_train)
    # acc = classifier.score(X_test, y_test)
    # print(acc)
    # y_pred = classifier.predict(X_test)
    # confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
    # print(confusion_matrix)


    # audio file

    # print(position_list)


