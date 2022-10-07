from sqlite3 import adapt


sr = 4000
mel_num = 128
hop_length = 63
frame_length = 628
num_classes = 3
location_dict = {'AV':0, 'MV':1, 'PV':2, 'TV':3}
outcome_dict = {'Normal':0, 'Abnormal':1}
murmur_dict = {'Absent':0, 'Present':1, 'Unknown':2}
adapt_learning_rate = False
# location_reverse_dict = {'AV':0, 'MV':1, 'PV':2, 'TV':3}


# training parameters =============================
model_name = 'CRNN_lr0001_1007_0418'
model_type = 'CRNN'
epochs = 200
batch_size = 12
learning_rate = 0.0001
