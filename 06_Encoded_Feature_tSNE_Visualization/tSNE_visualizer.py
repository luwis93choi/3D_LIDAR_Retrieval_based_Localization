import os
import sys
import argparse

from sklearn import cluster
import cv2 as cv
import numpy as np

import time
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import matplotlib.animation as animation

import collections

import pytorch_ssim     # SSIM Loss : https://github.com/Po-Hsun-Su/pytorch-ssim

import torch
from torch.utils.data import DataLoader

from sensor_dataset import sensor_dataset

from CNN_Autoencoder import CNN_Autoencoder

from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering

### Dataset & Model Preparation ###
ap = argparse.ArgumentParser()

ap.add_argument('-l', '--input_lidar_file_path', type=str, required=True)
ap.add_argument('-i', '--input_img_file_path', type=str, required=True)
ap.add_argument('-p', '--input_pose_file_path', type=str, required=True)
ap.add_argument('-b', '--input_batch_size', type=int, required=True)
ap.add_argument('-c', '--input_CUDA_num', type=str, required=True)
ap.add_argument('-e', '--training_epoch', type=int, required=True)
ap.add_argument('-m', '--trained_model', type=str, required=True)
ap.add_argument('-a', '--save_plot_animation', type=int, required=True)

args = vars(ap.parse_args())

batch_size = args['input_batch_size']

cuda_num = args['input_CUDA_num']

training_epoch = args['training_epoch']

model_path = args['trained_model']

save_plot_animation = args['save_plot_animation']

if cuda_num != '':        
    # Load main processing unit for neural network
    PROCESSOR = torch.device('cuda:'+cuda_num if torch.cuda.is_available() else 'cpu')

print('Device in use : {}'.format(PROCESSOR))

train_sequence=['00', '01', '05', '08', '09']
valid_sequence=['03', '04', '06', '07']
test_sequence=['02', '10']

dataset = sensor_dataset(lidar_dataset_path=args['input_lidar_file_path'], 
                        img_dataset_path=args['input_img_file_path'], 
                        pose_dataset_path=args['input_pose_file_path'],
                        train_sequence=train_sequence, valid_sequence=valid_sequence, test_sequence=test_sequence)

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
dataloader.dataset.mode = 'validation'

start_time = str(time.time())
loss_history = []

loss_Q = collections.deque(maxlen=1000)


### Feature Encoding using CNN-based Autoencoder ###

encoded_feature_list = []
xyz_pose_list = []

# for batch_idx, (lidar_range_img_tensor, current_img_tensor, pose_6DOF_tensor) in enumerate(dataloader):
for batch_idx, (current_img_tensor, pose_6DOF_tensor) in enumerate(dataloader):

    # combined_sensor_img_tensor = (torch.cat((lidar_range_img_tensor, current_img_tensor), dim=1)).to(PROCESSOR)
    combined_sensor_img_tensor = current_img_tensor.to(PROCESSOR)

    if batch_idx == 0:
        
        checkpoint = torch.load(model_path)
        
        Autoencoder = CNN_Autoencoder(device=PROCESSOR, input_size=combined_sensor_img_tensor.shape, batch_size=batch_size, learning_rate=0.001)
        Autoencoder.load_state_dict(checkpoint['Autoencoder'])
        Autoencoder.eval()

        print('Model loaded : {}'.format(model_path))

    encoded_feature_tensor = Autoencoder.encoder(combined_sensor_img_tensor)

    flat_encoded_feature_tensor = torch.flatten(encoded_feature_tensor, start_dim=1)

    flat_encoded_feature = flat_encoded_feature_tensor.clone().detach().cpu().numpy()[0]

    encoded_feature_list.append(flat_encoded_feature)
    xyz_pose_list.append(pose_6DOF_tensor.clone().cpu().numpy()[0, 0:3])

    est_img_tensor = Autoencoder.decoder(encoded_feature_tensor)

    recovery_loss = -Autoencoder.loss(est_img_tensor, combined_sensor_img_tensor)

    loss_Q.append(recovery_loss.item())

    updates = []
    updates.append('\n')
    updates.append('[Progress : {:.2%}][Batch Idx : {}] \n'.format(batch_idx/len(dataloader), batch_idx))
    # updates.append('[Batch Idx : {}] : {} \n'.format(batch_idx, flat_encoded_feature))
    updates.append('[Immediate Loss] : {:.4f} \n'.format(recovery_loss.item()))
    updates.append('[Running Average Loss] : {:.4f} \n'.format(sum(loss_Q) / len(loss_Q)))
    final_updates = ''.join(updates)

    sys.stdout.write(final_updates)

    if batch_idx < len(dataloader)-1:
        for line_num in range(len(updates)):
            sys.stdout.write("\x1b[1A\x1b[2K")

    if batch_idx == 0:
        if os.path.exists('./[Valid]' + start_time) == False:
            print('Creating save directory')
            os.mkdir('./[Valid]' + start_time)

    f = open('./[Valid]' + start_time + '/running_avg_loss.txt', 'a')
    f.write(str(sum(loss_Q) / len(loss_Q)) + '\n')
    f.close()

    f = open('./[Valid]' + start_time + '/immediate_avg_loss.txt', 'a')
    f.write(str(recovery_loss.item()) + '\n')
    f.close()


### Pose-based Agglomerative Clustering ###

agglo_clusterer = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold=5.0)
clusters = agglo_clusterer.fit(xyz_pose_list)

num_clusters = len(np.unique(clusters.labels_))

color_map = []
cmap = matplotlib.cm.get_cmap('rainbow')
for i in range(num_clusters):
    color_map.append(cmap(i/num_clusters))



### t-SNE Encoded Feature Visualization ###

fig = plt.figure(figsize=(20, 16))

encoded_feature_list = np.array(encoded_feature_list)

print(encoded_feature_list.shape)

embedded_encoded_features = TSNE(n_components=2, verbose=1, random_state=42, n_jobs=8).fit_transform(encoded_feature_list)

print(embedded_encoded_features.shape)

plt.title('CNN Autoencoder-based encoded feature visualization with t-SNE')
plt.xlabel('t-SNE embedded feature[0]')
plt.ylabel('t-SNE embedded feature[1]')
for idx in range(len(embedded_encoded_features)):
    print('[t-SNE Plotting Progress : {:.2%}]'.format(idx/len(embedded_encoded_features)))
    # print('{} {} : {}'.format(embedded_encoded_features[idx, 0], embedded_encoded_features[idx, 1], str(clusters.labels_[idx])))
    plt.scatter(embedded_encoded_features[idx, 0], embedded_encoded_features[idx, 1], color=color_map[clusters.labels_[idx]])
    plt.text(embedded_encoded_features[idx, 0], embedded_encoded_features[idx, 1], str(clusters.labels_[idx]), fontsize=6, color='black')
    
plt.tight_layout()
plt.show()

if save_plot_animation:
    plt.cla()
    
    current_idx = 0
    def draw_func(each_frame):

        global current_idx

        x, y = each_frame

        plt.scatter(x, y, color=color_map[clusters.labels_[current_idx]])
        plt.text(x, y, str(clusters.labels_[current_idx]), fontsize=12, color='black')

        current_idx += 1

    tSNE_animation = animation.FuncAnimation(fig=fig,
                                            func=draw_func,
                                            frames=embedded_encoded_features)

    writer = animation.writers['ffmpeg'](fps=25)
    tSNE_animation.save('./[Valid]' + start_time + '/tSNE_visualization_result.mp4', writer=writer, dpi=128)



    