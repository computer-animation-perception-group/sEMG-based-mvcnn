# Surface Electromyography-based Gesture Recognition by Multi-view Deep Learning
This repo contains the contains the code of our latest paper on sEMG: Wentao Wei,Qingfeng Dai,Yongkang Wong,Yu Du,Kankanhalli,Weidong Geng."[Surface Electromyography-based Gesture Recognition by Multi-view Deep Learning](https://ieeexplore.ieee.org/abstract/document/8641445/)"

## Requirements
- A CUDA compatible GPU
- Ubuntu >= 14.04 or any other Linux/Unix that can run Docker
- [Docker](http://docker.io/)
- [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker)

## Usage
- **Pull docker image**

    We have uploaded our docker image to the [docker hub](https://hub.docker.com/). You can pull the docker image using the command line as follows:
    ```
    docker pull zjucapg/semg:latest
    ```
    Use command line below to get into the docker container
    ```
    nvidia-docker run -ti -v your_prodjectdir:/code your_featuredir:/feature your_imudir:/imu zjucapg/semg /bin/bash
    ```
- **Dataset**

    The original datasets used in the paper consists of 11 categories including [NinaPro Databases](https://www.idiap.ch/project/ninapro/database)(**DB1-DB7**) and [BiopatRec Databases](https://github.com/biopatrec/biopatrec/wiki/Data_Repository.md)(**10mov4chUntargetedForearm, 6mov8chUFS, 10mov4chUF_AFEs, 8mov16chLowerLimb**).

    In this work, handcrafted features of sEMG are used as different views of multiview deep learning. Thus, we will do feature extraction for sEMG data.
    - **Segmentation**

        All the datasets metioned above need to be segmented by gesture labels and stored in Matlab format. Take NinaPro DB1 as an exmaple. `ninapro-db1/data/sss/ggg/sss_ggg_ttt.mat` contains a field `data` represents the trial `ttt` of gesture `ggg` of subject `sss`. And numbers start from zero. Gesture 0 is the rest gesture.
    - **Feature Extraction**

        After segmentation, we will do handcrafted feature extraction for all the datasets metioned above. The features we used in the paper are `Mean Absolute Value (MAV)`, `Waveform Length(WL)`, `Willison Amplitude(WAMP)`, `Zero Crossing(ZC)`, `Mean Absolute` `Value Slope(MAVS)`, `Autoregressive Coefficients(ARC)`, `Mean Frequency(MNF)`, `Power Spectrum Ratio(PSR)`, `Discrete Wavelet Transform Coefficients(DWTC)`, `Discrete Wavelet Packet Transform Coefficients(DWPTC)`. To accelerate training and testing, we save the feature data as .mat file. For example, `ninapro-db1/data/001/002/001_002_003_mav.mat` represents the `MAV` feature of 3th trial of 2rd gesture of 1st subject. There are two ways to access the feature datasets.

        One way is extracting features using the command line in the repo as follows
        ```
        cd extract_features
        # You can change the dataset you want to  extract in this script
        sh feature_extract.sh  
        ```
        The other way is to download prepared feature dataset on Google drive, which only contains features of NinaPro DB1 due to the huge amount of data.
    - **IMU data**

        We also use IMU as a type of different views. For the datasets contains IMU data such as NinaPro DB2, the IMU data should be segmented like sEMG data in the section Segmentation. And IMU data is stored as .mat file such as `ninapro-db2/imu/001/002/001_002_003.mat`. For this part, you have to make it by yourself.
- **Quick Start**
```
# Train singleview intra-subject without imu
./noimu-single-intra-subject.sh
# Train multiview intra-subject
./noimu-multi-intra-subject.sh
# Train singleview intra-subject with imu
./imu-single-intra-subject.sh
# Train multiview intra-subject with imu
./imu-multi-intra-subject.sh
# Test with parameters
python test_ninapro_db3-4-6-7.py -d ninapro-db1 -c one-fold-intra-subject --fusion single 
```
- **Trained Model**

We provide trained models on datasets [NinaPro DB1](https://drive.google.com/open?id=1sxbrq2ubGgLcrCwq-AUAauLkIuvACQ7-), [NinaPro DB2](https://drive.google.com/open?id=1aqjUFAhcZ1D2Jby6m8M6VLglXWiy-ePv) and [NinaPro DB5](https://drive.google.com/open?id=1_h4o6fLY3lajlSOVwkm-I56qBMcAqGSm) without imu data. We also provide [trained models](https://drive.google.com/open?id=1CzT3Xa6ktKr748OtuqI91sK8WdzLN-sN) with imu data on NinaPro DB5.
## License
Licensed under an GPL v3.0 license.

## Bibtex
```
@article{wei2019surface,
 title={Surface Electromyography-based Gesture Recognition by Multi-view Deep Learning},
 author={Wei, Wentao and Dai, Qingfeng and Wong, Yongkang and Hu, Yu and Kankanhalli, Mohan and Geng, Weidong},
 journal={IEEE Transactions on Biomedical Engineering},
 year={2019},
 publisher={IEEE}
}
```