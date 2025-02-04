import os
import json
import librosa
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ICBHI_Dataset(Dataset):
    def __init__(self, \
        diagnosis_csv, filelist_csv, data_dir, \
        sample_rate: int=16000, segment_length: int=1024, \
        n_mels: int=128, win_dur: float=0.025, hop_dur: float=0.01, \
        type_mode: str='train', \
        transform=None) -> None:
        filelist = pd.read_csv(filelist_csv)
        diagnosis = pd.read_csv(diagnosis_csv)
        self.segmetn_length = segment_length
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_dur = hop_dur
        self.win_dur = win_dur
        self.transform = transform
        
        self.segment_list = []
        # load everything into 1 table
        for idx, row in filelist.iterrows():
            file_id = row['file_id']
            if row['tag']!=type_mode:
                continue
            file_info = file_id.split('_')
            patient_id = int(file_info[0])
            recording_index = file_info[1]
            chest_location = file_info[2]
            channels = file_info[3]
            acq_device = file_info[4]
            # Save information into the filelist
            filelist.loc[idx, 'patient_id'] = patient_id
            filelist.loc[idx, 'recording_index'] = recording_index
            filelist.loc[idx, 'chest_location'] = chest_location
            filelist.loc[idx, 'channels'] = channels
            filelist.loc[idx, 'acq_device'] = acq_device
            # load the text file
            txt_file = os.path.join(self.data_dir, file_id + '.txt')
            meta_info = {'start': [], 'stop': [], 'crackles': [], 'wheezes': []}
            with open(txt_file, 'r') as f:
                contents = f.readlines()
                contents = [x.strip() for x in contents]
                # Save information into the filelist
                for line in contents:
                    line = line.split('\t')
                    meta_info['start'].append(float(line[0]))
                    meta_info['stop'].append(float(line[1]))
                    meta_info['crackles'].append(int(line[2]))
                    meta_info['wheezes'].append(int(line[3]))
            
            self.segment_hop = self.segmetn_length // 2
            n_segments = int((meta_info['stop'][-1] - (self.segmetn_length * self.hop_dur))//(self.segment_hop *self.hop_dur))
            n_frames = int((meta_info['stop'][-1] - self.win_dur) // self.hop_dur) + 1
            crackles_labels_frame = np.zeros((n_frames, 1))
            wheezes_labels_frame = np.zeros((n_frames, 1))
            # 4 classes: normal, crackles, wheezes, both = [0, 1, 2, 3]
            for start, stop, crackles, wheezes in zip(meta_info['start'], meta_info['stop'], meta_info['crackles'], meta_info['wheezes']):
                start_idx = int(start / self.hop_dur)
                stop_idx = int(stop / self.hop_dur)
                crackles_labels_frame[start_idx:stop_idx] = crackles
                wheezes_labels_frame[start_idx:stop_idx] = wheezes * 2
            
            for i in range(n_segments):
                start_point = i * self.segment_hop
                batch_label = int(max(crackles_labels_frame[start_point:start_point+self.segmetn_length]) + \
                    max(wheezes_labels_frame[start_point:start_point+self.segmetn_length]))
                self.segment_list.append({\
                    'file_id': file_id,\
                    'segment_id': i,\
                    'label': batch_label, \
                    'meta_info': meta_info})
                
            # Save information into the filelist
            filelist.loc[idx, 'meta_info'] = "{:s}".format(json.dumps(meta_info)) # save as string
        # join the diagnosis table with the filelist table
        self.filelist = pd.merge(filelist, diagnosis, on='patient_id')
        # drop rows which rag is type_mode
        self.filelist = self.filelist[self.filelist['tag']==type_mode]
        # filter the filelist for the mode
        self.mode = type_mode
        # save segment list to a json file
        with open('segment_list_' + type_mode + '.json', 'w') as f:
            json.dump(self.segment_list, f, indent=4)
        
    
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, idx):
        row = self.segment_list[idx]
        segment_id = row['segment_id']
        sr = self.sample_rate
        s, sr = librosa.load(os.path.join(self.data_dir, row['file_id'] + '.wav'), sr=16000)
        # meta_info = row['meta_info']
        if s.shape[0] < self.segmetn_length:
            s = np.pad(s, (0, self.segmetn_length - s.shape[0]), mode='constant')
        # extract Mel spectrogram
        mel_s = librosa.feature.melspectrogram(y=s, sr=sr, n_mels=self.n_mels, \
            hop_length=int(self.hop_dur*sr), win_length=int(self.win_dur*sr),center=False)
        mel_s = librosa.power_to_db(mel_s, ref=np.max)

        start_point = segment_id * self.segment_hop
        if self.transform:
            batch_mel_s = self.transform(mel_s[:,start_point:start_point+self.segmetn_length].T)
            batch_labels = row['label']

        return batch_mel_s, batch_labels
    
    # def __getitem__(self, idx):
    #     row = self.filelist.iloc[idx]
    #     s, sr = librosa.load(os.path.join(self.data_dir, row['file_id'] + '.wav'), sr=None)
    #     meta_info = json.loads(row['meta_info'])

    #     # extract Mel spectrogram
    #     mel_s = librosa.feature.melspectrogram(y=s, sr=sr, n_mels=self.n_mels, \
    #         hop_length=int(self.hop_dur*sr), win_length=int(self.win_dur*sr),center=False)
    #     mel_s = librosa.power_to_db(mel_s, ref=np.max)
    #     crackles_labels_frame = np.zeros((mel_s.shape[1], 1))
    #     wheezes_labels_frame = np.zeros((mel_s.shape[1], 1))
    #     # 4 classes: normal, crackles, wheezes, both = [0, 1, 2, 3]
    #     for start, stop, crackles, wheezes in zip(meta_info['start'], meta_info['stop'], meta_info['crackles'], meta_info['wheezes']):
    #         start_idx = int(start / self.hop_dur)
    #         stop_idx = int(stop / self.hop_dur)
    #         crackles_labels_frame[start_idx:stop_idx] = crackles
    #         wheezes_labels_frame[start_idx:stop_idx] = wheezes * 2

    #     segment_hop = self.segmetn_length // 2
    #     n_segments = (mel_s.shape[1] - self.segmetn_length) // segment_hop + 1
    #     batch_mel_s = torch.zeros((self.segmetn_length, self.n_mels))
    #     batch_labels = torch.zeros((1))
    #     # randomly pick a start point
    #     start_point = np.random.randint(0, mel_s.shape[1] - self.segmetn_length)
    #     if self.transform:
    #         batch_mel_s = self.transform(mel_s[:,start_point:start_point+self.segmetn_length].T)
    #         data = crackles_labels_frame[start_point:start_point+self.segmetn_length] + \
    #             wheezes_labels_frame[start_point:start_point+self.segmetn_length]
    #         batch_labels = np.argmax(np.bincount(data.flatten().astype(int)))
    #     # if self.transform:
    #     #     for i in range(n_segments):
    #     #         batch_mel_s[i,:,:] = self.transform(mel_s[:,i*segment_hop:i*segment_hop+self.segmetn_length].T)
    #     #         data = crackles_labels_frame[i*segment_hop:i*segment_hop+self.segmetn_length] + \
    #     #             wheezes_labels_frame[i*segment_hop:i*segment_hop+self.segmetn_length]
    #     #         batch_labels[i] = np.argmax(np.bincount(data.flatten().astype(int)))
    #     # print('data shape:', batch_mel_s.shape)
    #     return batch_mel_s, batch_labels
        