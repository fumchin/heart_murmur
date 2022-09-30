sr = 4000
mel_num = 128
hop_length = 128
frame_length = 628
num_classes = 4
location_dict = {'AV':0, 'MV':1, 'PV':2, 'TV':3}
outcome_dict = {'Normal':0, 'Abnormal':1}
# location_reverse_dict = {'AV':0, 'MV':1, 'PV':2, 'TV':3}


# training parameters =============================
model_name = 'CRNN_lr001_0921_200'
model_type = 'CRNN'
epochs = 200
batch_size = 12
learning_rate = 0.001
