# Joint-beat-and-downbeat-estimation

We propose a deep learning model for joint beat and downbeat estimation. We tackle the task without incorporating a postprocessing network (often dynamic Bayesian networks). By inspecting a state-of-the-art convolutional approach, we propose several reformulations regarding the network architecture and the loss function. For further details, please refer to ["Toward Postprocessing-free Neural Networks for Joint Beat and Downbeat Estimation" (ISMIR 2023)](https://).

## Model Architecture ##
<div style="width: 60%; height: 60%">
  
![image](https://github.com/Tsung-Ping/Joint-beat-and-downbeat-estimation/blob/main/image/architecture.png)
  
</div>


## Pre-trained Model
A model pre-trained on the Ballroom dataset (except tracks whose id=0) is provided. The id of each track can be found in [splits]( https://github.com/superbock/ISMIR2020/blob/master/splits/ballroom_8-fold_cv_dancestyle.folds).

## Requirements
 * python >= 3.6.9
 * tensorflow >= 2.5.0
 * numpy >= 1.19.5
 * mir_eval = 0.6