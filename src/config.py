sr = 4000
mel_num = 128
hop_length = 128
frame_length = 628
num_classes = 4
location_dict = {'AV':0, 'MV':1, 'PV':2, 'TV':3}

# training parameters =============================
model_name = 'crnn_fpn_test'
epochs = 100
batch_size = 24
learning_rate = 0.001
