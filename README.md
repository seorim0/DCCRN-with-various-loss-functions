# DCCRN with various loss functions

DCCRN(Deep Complex Convolutional Recurrent Network) is one of the deep neaural networks proposed at [[1]](https://arxiv.org/abs/2008.00264). This repository is an application using DCCRN with various loss functions. Our original paper can be found here, and you can check the test samples here.   
<br>   
   
![DCCRN_수정최종](https://user-images.githubusercontent.com/55497506/105969652-d39f6b80-60cb-11eb-805c-0f204405ef37.png)
> Source of the figure: (논문)   
<br>



# Loss functions
We use two base loss functions and two perceptual loss functions.

> Base
  1. MSE: Mean Squred Error   
  ![image](https://user-images.githubusercontent.com/55497506/106714015-97758900-663e-11eb-9593-6ecfd4266a41.png)
  <br>

  2. SI-SNR: Scale Invariant Source-to-Noise Ratio   
  ![image](https://user-images.githubusercontent.com/55497506/106714206-da376100-663e-11eb-94c6-77f6588616b9.png)
  <br>

> Perceptual
  1. LMS: Log Mel Spectra   
  ![image](https://user-images.githubusercontent.com/55497506/106714238-e58a8c80-663e-11eb-8601-58bb020a2d3b.png)
  <br>

  2. PMSQE: Perceptual Metric for Speech Quality Evaluation   
  ![image](https://user-images.githubusercontent.com/55497506/106714147-c855be00-663e-11eb-8a8d-a9d5aba1325d.png)
  <br>


# Requirements
This repository is tested on Ubuntu 20.04.
* Python 3.7+
* Cuda 10.0+
* CuDNN 7+
* Pytorch 

# Prepare training data
The data consists of the following three dimensions.   
```[Batch size, 2(input & target), wav length]```   
We use 3 seconds of wav files and the sampling frequency is 16k.

# Use pretrained models

# References
* [DCCRN](https://github.com/huyanxin/DeepComplexCRN)

# Paper
Performance comparison evaluation of speech enhancement using various loss function.
