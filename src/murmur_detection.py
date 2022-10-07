from importlib.metadata import PathDistribution
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

matplotlib.use('Agg')


def preprocess(audio_file_list, saved_path, sr, mel_num, hop_length):
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
            np.save(os.path.join(saved_path, file_name+'_'+str(frame_count)+'.npy'), S_scaled)
        
        # S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_num, hop_length=hop_length)
        # print(S.shape)
        # scaler = MinMaxScaler(feature_range=(0, 1)).fit(S)
        # scaler = StandardScaler().fit(S)
        # S_scaled = scaler.transform(S)
        # file_name = os.path.splitext(os.path.basename(audio_file))[0]
        # # save feature
        # np.save(os.path.join(saved_path, file_name+'.npy'), S_scaled)


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    return train_dataloader
def change_learning_rate(optimizer, epochs):
    pass
def train(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device, epochs, checkpoint_path):

    val_loss_min = math.inf
    f1_max = 0
    training_loss_list = []
    validation_loss_list = []
    training_f1_list = []
    validation_f1_list = []
    epoch_list = []
    for i in range(epochs):
        print(f"Epoch {i+1}")
        epoch_list.append(i)
        # change learning rate
        if cfg.adapt_learning_rate == True:
            change_learning_rate(optimizer, epochs)
        training_loss, validation_loss = train_single_epoch(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device)

        training_loss_list.append(training_loss)
        validation_loss_list.append(validation_loss)
        
        # training_f1_list.append(train_f1_score)
        # validation_f1_list.append(val_f1_score)
        
        model.train()
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_path, f'checkpoint_epoch_{i}.pt'))

        # if validation_loss < loss_min:
        if validation_loss < val_loss_min:
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_path, f'best_model.pt'))

            # loss_min = validation_loss
            val_loss_min = validation_loss

            output_txt = os.path.join(checkpoint_path, 'best.txt')
            with open(output_txt, 'w') as f:
                f.write(f'training_loss: {training_loss}\n')
                f.write(f'validation_loss: {validation_loss}\n')
                # f.write(f'training_f1: {train_f1_score}\n')
                # f.write(f'validation_f1: {val_f1_score}\n')
                f.write(f'epoch: {i}\n')

        print("---------------------------")
        if (i+1) % 5 == 0:
            # draw training & validation loss
            plt.figure(1)
            plt.grid()
            # plt.xlim(0, 200, 1)
            plt.plot(epoch_list, training_loss_list, 'b')
            plt.plot(epoch_list, validation_loss_list, 'r')
            plt.savefig(os.path.join(checkpoint_path, f'{cfg.model_name}_loss.png'))
            
            # plt.figure(2)
            # plt.grid()
            # # plt.xlim(0, 200, 1)
            # plt.plot(epoch_list, training_f1_list, 'b')
            # # plt.plot(epoch_list, validation_f1_list, 'r')
            # plt.savefig(os.path.join(checkpoint_path, f'{cfg.model_name}_f1.png'))

    print("Finished training")
    plt.figure(1)
    plt.grid()
    plt.plot(epoch_list, training_loss_list, 'b')
    plt.plot(epoch_list, validation_loss_list, 'r')
    plt.savefig(os.path.join(checkpoint_path, f'{cfg.model_name}_loss.png'))

    # plt.figure(2)
    # plt.grid()
    # plt.plot(epoch_list, training_f1_list, 'b')
    # # plt.plot(epoch_list, validation_f1_list, 'r')
    # plt.savefig(os.path.join(checkpoint_path, f'{cfg.model_name}_f1.png'))



def train_single_epoch(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device):
    
    training_seg_loss = 0
    training_patient_loss = 0
    training_loss = 0
    
    pred_array = []
    ground_array = []
    weight = torch.Tensor([1, 5, 3]).to(device)
    model.train()
    patient_target_dict = {}
    patient_pred_dict = {}
    for count, (input, target, patient_id_list) in enumerate(train_dataloader):
        input, target = input.to(device), target.to(device)
        # calculate loss
        prediction = model(input)
        target_onehot = nn.functional.one_hot(target, num_classes=cfg.num_classes)
        y_pred = torch.argmax(prediction, dim=1)
        # print('input: ', input)
        # print('target: ', target)
        # print('prediction: ', prediction)
        # loss_seg = loss_fn(prediction, target_onehot.float())
        # loss_seg = loss_seg * weight
        # loss_seg = loss_seg.mean()
        loss_seg = loss_fn(prediction, target)
        training_seg_loss += loss_seg.item()
        optimizer.zero_grad()
        loss_seg.backward()
        optimizer.step()
        
        if count == 0:
                pred_array = y_pred.detach().cpu().numpy()
                ground_array = target.cpu().numpy()
        else:
            pred_array = np.concatenate((pred_array, y_pred.detach().cpu().numpy()), axis=None)
            ground_array = np.concatenate((ground_array, target.cpu().numpy()), axis=None)
        # put into list
        
        
    
        
    
    # train_f1_score = f1_score(ground_array, pred_array, average='macro')
    patient_train_loss = 0
    patient_id_batch_list = []
    for count, (input, target, patient_id_list) in enumerate(train_dataloader):
        
        input, target = input.to(device), target.to(device)
        target_onehot = nn.functional.one_hot(target, num_classes=cfg.num_classes)
        prediction = model(input)
        for patient_count, patient_id in enumerate(patient_id_list):
            patient_target_dict[patient_id] = target[patient_count]
            try:
                patient_pred_dict[patient_id] = torch.add(patient_pred_dict[patient_id], prediction[patient_count])
            except:
                patient_pred_dict[patient_id] = prediction[patient_count]
                patient_id_batch_list.append(patient_id)
        if ((count + 1) % 5 == 0 ) or (count+1) == len(train_dataloader):
            for patient_count, patient_id in enumerate(patient_id_batch_list):
                if patient_count == 0:
                    patient_train_loss = loss_fn(patient_pred_dict[patient_id], patient_target_dict[patient_id])
                else:
                    patient_train_loss = patient_train_loss + loss_fn(patient_pred_dict[patient_id], patient_target_dict[patient_id])
            # training_patient_loss += patient_train_loss.mean().item()
            # patient_train_loss = nn.Softmax.apply(patient_train_loss)
            # patient_train_loss = patient_train_loss * weight
            # patient_train_loss = patient_train_loss.mean()
            training_patient_loss += patient_train_loss.item()
            optimizer.zero_grad()
            patient_train_loss.backward()
            optimizer.step()
            
            patient_id_batch_list = []
            patient_pred_dict = {}
            patient_target_dict = {}
            patient_train_loss = 0
                
                

    
    training_loss = (training_patient_loss) / len(train_dataloader)
    # print(ABSENT_COUNT, PRESENT_COUNT, UNKNOWN_COUNT)
    print(f"loss: {training_loss}")
        
    model.eval()
    validation_loss = 0
    pred_array_val = []
    ground_array_val = []
    patient_val_loss = 0
    patient_id_batch_list = []
    patient_pred_dict = {}
    patient_target_dict = {}
    with torch.no_grad():
        for count, (input, target, patient_id_list) in enumerate(validation_dataloader):
            
            input, target = input.to(device), target.to(device)
            target_onehot = nn.functional.one_hot(target, num_classes=cfg.num_classes)
            prediction = model(input)
            for patient_count, patient_id in enumerate(patient_id_list):
                patient_target_dict[patient_id] = target[patient_count]
                try:
                    patient_pred_dict[patient_id] = torch.add(patient_pred_dict[patient_id], prediction[patient_count])
                except:
                    patient_pred_dict[patient_id] = prediction[patient_count]
                    patient_id_batch_list.append(patient_id)
            if ((count + 1) % cfg.batch_size == 0 ) or (count+1) == len(validation_dataloader):
                for patient_count, patient_id in enumerate(patient_id_batch_list):
                    if patient_count == 0:
                        patient_val_loss = loss_fn(patient_pred_dict[patient_id], patient_target_dict[patient_id])
                    else:
                        patient_val_loss = patient_val_loss + loss_fn(patient_pred_dict[patient_id], patient_target_dict[patient_id])
                # training_patient_loss += patient_train_loss.mean().item()
                # patient_val_loss = patient_val_loss * weight
                # patient_val_loss = patient_val_loss.mean()
                validation_loss += patient_val_loss.item()
                
                patient_id_batch_list = []
                patient_pred_dict = {}
                patient_target_dict = {}
                patient_val_loss = 0
        # for count, (features, label, patient_id_list) in enumerate(validation_dataloader):
        # # for count, (features, label) in enumerate(train_dataloader):

        #     features = features.to(device)
        #     label = label.to(device)
        #     label_onehot = nn.functional.one_hot(label, num_classes=cfg.num_classes)
        #     pred = model(features)
        #     y_pred = torch.argmax(pred, dim=1)
           
        #     val_loss = loss_fn(pred, label_onehot.float())
        #     val_loss = val_loss * weight
        #     val_loss = val_loss.mean()
        #     validation_loss += val_loss.item()

        #     if count == 0:
        #         pred_array_val = y_pred.detach().cpu().numpy()
        #         ground_array_val = label.cpu().numpy()
        #     else:
        #         pred_array_val = np.concatenate((pred_array_val, y_pred.detach().cpu().numpy()), axis=None)
        #         ground_array_val = np.concatenate((ground_array_val, label.cpu().numpy()), axis=None)
    
    # for count,(input, target) in enumerate(validation_dataloader):
    #     input, target = input.to(device), target.to(device)

    #     # calculate loss
    #     prediction = model(input)
    #     y_pred = torch.argmax(prediction, dim=1)
    #     val_loss = loss_fn(prediction, target)
    #     validation_loss += val_loss.item()
        
    #     if count == 0:
    #             pred_array = y_pred.detach().cpu().numpy()
    #             ground_array = target.cpu().numpy()
    #     else:
    #         pred_array = np.concatenate((pred_array, y_pred.detach().cpu().numpy()), axis=None)
    #         ground_array = np.concatenate((ground_array, target.cpu().numpy()), axis=None)
            
    validation_loss = validation_loss / len(validation_dataloader)
    print(f"val_loss: {validation_loss}")
    # val_f1_score = f1_score(ground_array_val, pred_array_val, average='macro')
    # print('f1 score: ' + str(val_f1_score))
    # print(confusion_matrix(ground_array_val, pred_array_val))

    return training_loss, validation_loss





if __name__ == '__main__':
    dataset_name = 'the-circor-digiscope-phonocardiogram-dataset-1.0.3'
    data_path = os.path.join('..', 'dataset', dataset_name, 'training_data')
    audio_file_list = glob(os.path.join(data_path, '*.wav'))
    preprocess_path = os.path.join('..', 'dataset', dataset_name, 'training_preprocess')



    # audio preprocess
    if not os.path.exists(os.path.join(preprocess_path)):
        os.makedirs(preprocess_path)

        preprocess(audio_file_list = audio_file_list, saved_path = preprocess_path, sr = cfg.sr, mel_num=cfg.mel_num, hop_length=cfg.hop_length)
        feature_file_list = glob(os.path.join(preprocess_path, '*.npy'))
    else:
        feature_file_list = glob(os.path.join(preprocess_path, '*.npy'))
    # preprocess(audio_file_list = audio_file_list, saved_path = preprocess_path, sr = cfg.sr, mel_num=cfg.mel_num, hop_length=cfg.hop_length)
    # feature_file_list = glob(os.path.join(preprocess_path, '*.npy'))

    checkpoint_path = os.path.join('./stored_data/classification', cfg.model_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    dataset = PhonocardiogramDataset(preprocess_path, data_path)
    train_size = int(len(dataset) * 0.95)
    test_size = len(dataset) - train_size
    torch.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # validation_size = int(len(test_dataset) * 0.5)
    # test_size = len(test_dataset) - validation_size
    # test_dataset, validation_dataset = torch.utils.data.random_split(test_dataset, [test_size, validation_size])

    train_dataloader = create_data_loader(train_dataset, cfg.batch_size)
    # validation_dataloader = create_data_loader(validation_dataset, cfg.batch_size)
    test_dataloader = create_data_loader(test_dataset, cfg.batch_size)

    # =========================================================================
    # train
    # =========================================================================
    if cfg.model_type == 'CNN':
        model = Cnn_Test().to(device)
    elif cfg.model_type == 'CRNN_fpn':
        model = CRNN_fpn(1, cfg.num_classes, kernel_size=7 * [3], padding=7 * [1], stride=7 * [1], nb_filters=[16,  32,  64,  128,  128, 128, 128],
                attention=True, activation="GLU", dropout=0.5, n_RNN_cell=128, n_layers_RNN=2,
                pooling=[[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]).to(device)
    elif cfg.model_type == 'CRNN':
        model = CRNN(1, cfg.num_classes, kernel_size=7 * [3], padding=7 * [1], stride=7 * [1], nb_filters=[16,  32,  64,  128,  128, 128, 128],
            attention=True, activation="GLU", dropout=0.5, n_RNN_cell=128, n_layers_RNN=2,
            pooling=[[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]).to(device)


    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999))
    model.train()
    train(model=model, train_dataloader=train_dataloader, validation_dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device, epochs=cfg.epochs, checkpoint_path=checkpoint_path)
    #=========================================================================
    # testing
    # =========================================================================
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

    best_checkpoint = torch.load(os.path.join(checkpoint_path, 'best_model.pt'))
    model_eval.load_state_dict(best_checkpoint['model_state_dict'])
    model_eval.eval()

    pred_array = []
    ground_array = []
    with torch.no_grad():
        for count, (features, label) in enumerate(test_dataloader):
        # for count, (features, label) in enumerate(train_dataloader):

            features = features.to(device)
            label = label.to(device)
            pred = model_eval(features)
            y_pred = torch.argmax(pred, dim=1)

            if count == 0:
                pred_array = y_pred.detach().cpu().numpy()
                ground_array = label.cpu().numpy()
            else:
                pred_array = np.concatenate((pred_array, y_pred.detach().cpu().numpy()), axis=None)
                ground_array = np.concatenate((ground_array, label.cpu().numpy()), axis=None)
            # ground_array.append(label.cpu().numpy())
            # pred_array.append(y_pred.detach().cpu().numpy())

    print(confusion_matrix(ground_array, pred_array))
    print('f1 score: ' + str(f1_score(ground_array, pred_array, average='macro')))
    print('balanced acc: ' + str(balanced_accuracy_score(ground_array, pred_array)))
    print('end evaluation')

