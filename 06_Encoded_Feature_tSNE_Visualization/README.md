## Introduction
This is PyTorch-based implementation for training DNN-based encoder and visualizing its latent feature vector clustering.

This implementation can compare the latent feature vector clustering between DNN encoder trained with MSE loss and DNN encoder trained with MSE-based triplet loss.

This implementation utilizes t-SNE (T-distributed Stochastic Neighbor Embedding) for latente feature clustering visualization.

[PyQtGraph](https://www.pyqtgraph.org/) is used as t-SNE visualization tool.

## Requirements
[KITTI Visual Odometry Dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) (Dataset for training DNN endocer and visualizing latent feature vector clustering)

[PyTorch](https://pytorch.org/) (Higher than 1.9.0)

[PyQtGraph](https://www.pyqtgraph.org/)

[matplotlib](https://matplotlib.org/)

## Dataset Installation and Setup

- Download camera image, 3D LiDAR, 6DOF groundtruth dataset from [KITTI Visual Odometry Dataset website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- Set up the dataset directory as below

<table><tr><td valign="top" width="25%">
    
<div align="">

```
.
└── data_odometry_color
    └── dataset
        └── sequences
            ├── 00
            │   ├── image_2
            │   └── image_3
            ├── 01
            │   ├── image_2
            │   └── image_3
            ├── 02
            │   ├── image_2
            │   └── image_3
            ├── 03
            │   ├── image_2
            │   └── image_3
            ├── 04
            │   ├── image_2
            │   └── image_3
            ├── 05
            │   ├── image_2
            │   └── image_3
            ├── 06
            │   ├── image_2
            │   └── image_3
            ├── 07
            │   ├── image_2
            │   └── image_3
            ├── 08
            │   ├── image_2
            │   └── image_3
            ├── 09
            │   ├── image_2
            │   └── image_3
            └── 10
                ├── image_2
                └── image_3
```

</div>

</td><td valign="top" width="25%">
   
<div align="">

```
.
└── data_odometry_velodyne
    └── dataset
        └── sequences
            ├── 00
            │   └── velodyne
            ├── 01
            │   └── velodyne
            ├── 02
            │   └── velodyne
            ├── 03
            │   └── velodyne
            ├── 04
            │   └── velodyne
            ├── 05
            │   └── velodyne
            ├── 06
            │   └── velodyne
            ├── 07
            │   └── velodyne
            ├── 08
            │   └── velodyne
            ├── 09
            │   └── velodyne
            └── 10
                └── velodyne
```

</div>

</td><td valign="top" width="25%">
  
<div align="">

```
.
└── data_odometry_poses
    └── dataset
        └── poses
            ├── 00.txt
            ├── 01.txt
            ├── 02.txt
            ├── 03.txt
            ├── 04.txt
            ├── 05.txt
            ├── 06.txt
            ├── 07.txt
            ├── 08.txt
            ├── 09.txt
            └── 10.txt
```
</div>
  
</td></tr></table>  

## Running the Code

#### 1. DNN Encoder with MSE Loss

Specify following parameters in training execution shell script (execute.sh)

- lidar_dataset_path : Path to 3D LiDAR point cloud dataset directory
- image_dataset_path : Path to front camera image dataset directory
- pose_dataset_path : Path to 6DOF IMU groundtruth dataset directory
- batch_size : Batch size for training deep neural network
- cuda_num : Index number of CUDA to use
- training_epoch : Number of epochs to run for training

Train DNN by running training execution shell script 

```bash
./execute.sh
```

After completing the training, specify following parameters in t-SNE visualization shell script (execute_tSNE.sh)

- lidar_dataset_path : Path to 3D LiDAR point cloud dataset directory
- image_dataset_path : Path to front camera image dataset directory
- pose_dataset_path : Path to 6DOF IMU groundtruth dataset directory
- trained_model : Path to trained model produced from training execution shell script
- batch_size : Batch size for training deep neural network (* For t-SNE visualization, batch size must be set as 1.)
- cuda_num : Index number of CUDA to use
- training_epoch : Number of epochs to run for training

Run t-SNE visualization shell script to check latent feature vector clustering

```bash
./execute_tSNE.sh
```

#### 2. DNN Encoder with MSE-based Triplet Loss

Specify following parameters in execution shell script for triplet loss-based training and visualization (execute_tSNE_triplet.sh)

Set mode as 'training' in order to train DNN encoder with MSE-based triplet loss

Set mode as 'tsne' in order to visualize the latent feature vector clustering from trained network

- lidar_dataset_path : Path to 3D LiDAR point cloud dataset directory
- pose_dataset_path : Path to 6DOF IMU groundtruth dataset directory
- pretrained_model_path : Path to trained model produced from training mode
- batch_size : Batch size for training deep neural network (* For t-SNE visualization, batch size must be set as 1.)
- cuda_num : Index number of CUDA to use
- training_epoch : Number of epochs to run for training

Run execution shell script either for training or t-SNE visualization

```bash
./execute_tSNE_triplet.sh
```

## Further Customization

#### Switching to SSIM-based Loss

This implementation contains SSIM loss API implemented by [Po-Hsun-Su](https://github.com/Po-Hsun-Su/pytorch-ssim).

In order to set the network to train with SSIM loss, the network has to be declared with 'ssim' as its 'loss_type' argument.

```python
Feature_encoder = CNN_Encoder(device=PROCESSOR, input_size=anchor_img_tensor.shape, batch_size=batch_size, learning_rate=0.001, loss_type='ssim')

Autoencoder = CNN_Autoencoder(device=PROCESSOR, input_size=combined_sensor_img_tensor.shape, batch_size=batch_size, learning_rate=0.001, loss_type='ssim')
```
