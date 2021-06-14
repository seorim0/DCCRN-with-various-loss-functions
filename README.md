# DCCRN with various loss functions

DCCRN(Deep Complex Convolutional Recurrent Network) is one of the deep neaural networks proposed at [[1]](https://arxiv.org/abs/2008.00264). This repository is an application using DCCRN with various loss functions. Our original paper can be found [here](https://www.jask.or.kr/articles/xml/ABxn/), and you can check test samples [here](https://github.com/seorim0/DCCRN-with-various-loss-functions/tree/main/samples/0dB). Test samples are randomly choosed and we uploaded samples about SI-SNR and SI-SNR+LMS.    
<br>   
   
![DCCRN_수정최종](https://user-images.githubusercontent.com/55497506/105969652-d39f6b80-60cb-11eb-805c-0f204405ef37.png)
> Source of the figure: [paper](https://www.jask.or.kr/articles/xml/ABxn/)   
<br>



# Loss functions
We use two base loss functions and two perceptual loss functions.

> Base loss
  1. MSE: Mean Squred Error   
  ![image](https://user-images.githubusercontent.com/55497506/106714015-97758900-663e-11eb-9593-6ecfd4266a41.png)
  <br>

  2. SI-SNR: Scale Invariant Source-to-Noise Ratio   
  ![image](https://user-images.githubusercontent.com/55497506/106714206-da376100-663e-11eb-94c6-77f6588616b9.png)
  <br>

> Perceptual loss
  1. LMS: Log Mel Spectra   
  ![image](https://user-images.githubusercontent.com/55497506/106714238-e58a8c80-663e-11eb-8601-58bb020a2d3b.png)
  <br>

  2. PMSQE: Perceptual Metric for Speech Quality Evaluation   
  ![image](https://user-images.githubusercontent.com/55497506/106714147-c855be00-663e-11eb-8a8d-a9d5aba1325d.png)
  <br>   

We combined 2 types of base loss functons and 2 types of perceptual loss functions. The coupling constant ratio was determined experimentally. For example, in the case of MSE, which is the basic loss function, the initial size is about 0.001 ~ 0.002, whereas the LMS has an initial size of 0.1 ~ 0.2 and PMSQE is about 0.8 ~ 1.3. Therefore, to combine the two terms to be of similar size, a smaller coefficient was used in the perceptual based loss function term. The coupling constant ratio is a result of reflecting the dynamic range of the two terms rather than reflecting the sensitivity of the two terms. Meanwhile, in the course of the experiment, we determined that the basic loss function is a more important term, so we changed the coefficients so that the dynamic range ratio including the coupling constant could be adjusted from 1:1 to 10:1, respectively.   
 <br>
 
# Requirements
> This repository is tested on Ubuntu 20.04.
* Python 3.7+
* Cuda 10.1+
* CuDNN 7+
* Pytorch 1.7+
<br>

> Library
* tqdm
* asteroid
* scipy
* matplotlib
* tensorboardX

# Prepare data
The training and validation data consist of the following three dimensions.   
```[Batch size, 2(input & target), wav length]```   
<br>   
The test data consists of the following dimensions.   
```[noise type, dB classes, Batch size, 2(input & target), wav length]```   
We use 2 type of noise, seen and unseen and 7 dB classes from -10dB to 20dB.

<br>
We cut the wav files longer than 3 seconds into 3 seconds and zero padded for wav files shorter than 3 seconds.   
The sampling frequency is 16k.

<!--# Use pretrained models
If you want to test the model described in the [paper](), you can change chkpt_model path in ```config.py``` like ```'SI-SNR/'```  
<br>
We have uploaded 3 models trained with each loss function, SI-SNR, SI-SNR + LMS and SI-SNR + PMSQE.-->   

# Performance comparative evaluation
**Objective evaluation**   
<br>   
We evaluate the outputs with PESQ(Perceptual Evaluation of Speech Quality) and STOI(Short Time Objective Intelligibility measure).   
![t1](https://user-images.githubusercontent.com/55497506/108797149-e1aeb200-75cd-11eb-8ea4-3db00da21991.png)   
<br>   

![t2](https://user-images.githubusercontent.com/55497506/108797168-eb381a00-75cd-11eb-94ba-1d3a1016fb6e.png)   
<br>   

**Spectrogram**   

![image](https://user-images.githubusercontent.com/55497506/108705017-1a0fab00-7550-11eb-962a-9f0b218371a8.png)   
> Source of the figure: [paper]()   

The spectrograms of  (a) clean speech, (b) noisy speech at 0 dB SNR, estimated speeches using (c)  MSE and PMSQE, (d)  SI-SNR , (e) SI-SNR and PMSQE, (f)  SI-SNR and LMS. 

# References
**DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement**   
Yanxin Hu, Yun Liu, Shubo Lv, Mengtao Xing, Shimin Zhang, Yihui Fu, Jian Wu, Bihong Zhang, Lei Xie   
[[arXiv]](https://arxiv.org/abs/2008.00264)  [[code]](https://github.com/huyanxin/DeepComplexCRN)


# Paper
**Performance comparison evaluation of speech enhancement using various loss function.**   
Seo-Rim Hwang, Joon Byun, Young-Cheul Park   
[[paper]](https://www.jask.or.kr/articles/xml/ABxn/)   
   
   
* New version is my new clean-up code. I'm trying to the codes more clearly.   
* It's still in the editing phase. Please refer to the existing code. 
