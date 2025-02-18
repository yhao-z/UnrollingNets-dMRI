# UnrollingNets-dMRI
( If you find this code is helpful to you, could you please kindly give me a star <img src="https://slackmojis.com/emojis/13058-star_spin/download" width="30"/>)

This repo provides the TensorFlow 2.x implementation for several SOTA unrolling networks in accelerated MRI reconstruction.

It contains JotlasNet (MRI 2025), T2LR_Net (CIBM 2024), L+S_Net (MedIA 2021), SLR_Net (TMI 2021), ISTA_Net (CVPR 2018), DCCNN (TMI 2017).

| Method               | Paper                                                        | GitHub                                                    |
| -------------------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| JotlasNet (MRI 2025) | [Link](https://www.sciencedirect.com/science/article/pii/S0730725X25000190) | [Link](https://github.com/yhao-z/JotlasNet)               |
| T2LR_Net (CIBM 2024) | [Link](https://www.sciencedirect.com/science/article/pii/S0010482524001185) | [Link](https://github.com/yhao-z/T2LR-Net)                |
| L+S_Net (MedIA 2021) | [Link](https://www.sciencedirect.com/science/article/pii/S136184152100236X) | [Link](https://github.com/wenqihuang/LS-Net-Dynamic-MRI)  |
| SLR_Net (TMI 2021)   | [Link](https://ieeexplore.ieee.org/abstract/document/9481108) | [Link](https://github.com/Keziwen/SLR-Net)                |
| ISTA_Net (CVPR 2018) | [Link](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_ISTA-Net_Interpretable_Optimization-Inspired_CVPR_2018_paper.html) | [Link](https://github.com/jianzhangcs/ISTA-Net)           |
| DCCNN (TMI 2017)     | [Link](https://ieeexplore.ieee.org/abstract/document/8067520) | [Link](https://github.com/js3611/Deep-MRI-Reconstruction) |

## 1. Getting Started

### Environment Configuration

- we recommend to use docker

  ```shell
  # pull the docker images
  docker pull yhaoz/tf:2.9.0-bart
  # then you can create a container to run the code, see docker documents for more details
  ```

- if you don't have docker, you can still configure it via installing the requirements by yourself

  ```shell
  pip install -r requirements.txt # tensorflow is gpu version
  ```

Note that, we only run the code in NVIDIA GPU. In our implementation, the code can run normally in both Linux & Windows system.

### Dataset preparation

Actually, two dataset involves in this code, i.e., [OCMR](https://ocmr.info/) and [CMRxRecon](https://github.com/CmrxRecon/CMRxRecon). Here we just provides our pre-processing pipeline for OCMR dataset. Since CMRxRecon is a MICCAI challenge and has its own leader board, we do not provide our own code.

You could find the the single-coil and multi-coil dataset pre-processing and creating code in [yhao-z/ocmr-preproc-tf](https://github.com/yhao-z/ocmr-preproc-tf). You may need to put the pre-processed data into a file folder that contains four subfolders.

```shell
# the data needs to be arranged into four sub file folders, and you may set the datadir in the code.
- train
- val
- test
- masks
```

## 2. Run the code

### Test only

We provide the training weights on **OCMR** dataset for many sampling cases and both single-coil and multi-coil scenarios as in `weights-ocmr`. Note that the provided weights are only applicable in our data pre-processing implementation. **If you are using other different configuration, retraining from scratch is needed.**

```shell
# Please refer to main.py for more configurations.
python main.py --mode 'test'
```

### Training

```shell
# Please refer to main.py for more configurations.
python main.py --mode 'train'
```

## 3. To Do

- Add VISTA mask generation code.
- OCMR has release more data in last three years, but this code still uses the initial released limited data. Add more training data or make a new dataset in the future.

## 4. Note

- If this code have any error, feel free to raise an issue.
- If you have built your own model with this code, welcome to pull request and merge your model into this ModelZoo, of course,  if you want.
