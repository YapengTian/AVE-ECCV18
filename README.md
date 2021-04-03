Audio-Visual Event Localization in Unconstrained Videos (To appear in ECCV 2018) 

[Project](https://sites.google.com/view/audiovisualresearch) [ArXiv](https://arxiv.org/abs/1803.08842) [Demo Video](https://www.youtube.com/watch?v=m6r6BbD5MSc) 

[![Watch the video](Figs/demo_thumbnail.png)](https://www.youtube.com/watch?v=m6r6BbD5MSc)

### AVE Dataset & Features

AVE dataset can be downloaded from https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK.

[Audio feature](https://drive.google.com/file/d/1F6p4BAOY-i0fDXUOhG7xHuw_fnO5exBS/view?usp=sharing) and [visual feature](https://drive.google.com/file/d/1hQwbhutA3fQturduRnHMyfRqdrRHgmC9/view?usp=sharing) (7.7GB) are also released. Please put videos of AVE dataset into /data/AVE folder and features into /data folder before running the code. 

Scripts for generating audio and visual features: https://drive.google.com/file/d/1TJL3cIpZsPHGVAdMgyr43u_vlsxcghKY/view?usp=sharing (Feel free to modify and use it to precess your audio and visual data).

### Requirements

Python-3.6, Pytorch-0.3.0, Keras, ffmpeg. 

### Visualize attention maps

Run: python attention_visualization.py to generate audio-guided visual attention maps. 

![image](https://github.com/YapengTian/AVE-ECCV18/blob/master/Figs/att_easy.jpg)

### Supervised audio-visual event localization

Testing: 

A+V-att model in the paper: python supervised_main.py --model_name AV_att

[DMRN](https://drive.google.com/file/d/1D6M6lnUkS4yby0Y4LODIYAQUR6N_GtaR/view?usp=sharing) model in the paper:    python supervised_main.py --model_name DMRN  

Training:

python supervised_main.py --model_name AV_att --train



### Weakly-supervised audio-visual event localization
We add some videos without audio-visual events into training data. Therefore, the labels of these videos are background. Processed visual features can be found in [visual_feature_noisy.h5](https://drive.google.com/file/d/1I3OtOHJ8G1-v5G2dHIGCfevHQPn-QyLh/view?usp=sharing). Put the feature into data folder.

Testing: 

W-A+V-att model in the paper: python weak_supervised_main.py

Training:

python weak_supervised_main.py --train

### Cross-modality localization
For this task, we developed a cross-modal matching network. Here, we used visual feature vectors via global average pooling, and you can find [here](https://drive.google.com/file/d/1l-c8Kpr5SZ37h-NpL7o9u8YXBNVlX_Si/view?usp=sharing). Please put the feature into data folder. Note that the code was implemented via Keras-2.0 with Tensorflow as the backend.

Testing: 

python cmm_test.py

Training:

python cmm_train.py

### Other Related or Follow-up works

[1] Rouditchenko, Andrew, et al. "Self-supervised Audio-visual Co-segmentation." ICASSP, 2019. [[Paper]](https://ieeexplore.ieee.org/abstract/document/8682467)

[2] Lin, Yan-Bo, Yu-Jhe Li, and Yu-Chiang Frank Wang. "Dual-modality seq2seq network for audio-visual event localization." ICASSP, 2019 [[Paper]](https://arxiv.org/abs/1902.07473)

[3] Rana, Aakanksha, Cagri Ozcinar, and Aljosa Smolic. "Towards Generating Ambisonics Using Audio-visual Cue for Virtual Reality." ICASSP, 2019. [[Paper]](https://www.researchgate.net/profile/Cagri_Ozcinar/publication/332790611_Towards_Generating_Ambisonics_Using_Audio-visual_Cue_for_Virtual_Reality/links/5ccb031da6fdcc4719835ad3/Towards-Generating-Ambisonics-Using-Audio-visual-Cue-for-Virtual-Reality.pdf)

[4] Yu Wu, Linchao Zhu, Yan Yan, Yi Yang. "Dual Attention Matching for Audio-Visual Event Localization", ICCV, 2019. (oral) [[website]](https://yu-wu.net/)

[5] Jinxing Zhou, Liang Zheng, Yiran Zhong, Shijie Hao, Meng Wang. Positive Sample Propagation along the Audio-Visual Event Line, CVPR 2021. [[paper]](https://arxiv.org/abs/2104.00239)[[code]](https://github.com/jasongief/PSP_CVPR_2021)



### Citation

If you find this work useful, please consider citing it.

<pre><code>@InProceedings{tian2018ave,
  author={Tian, Yapeng and Shi, Jing and Li, Bochen and Duan, Zhiyao and Xu, Chenliang},
  title={Audio-Visual Event Localization in Unconstrained Videos},
  booktitle = {ECCV},
  year = {2018}
}
</code></pre>
 
 ### Acknowledgements
 
Audio features are extracted using [vggish](https://github.com/tensorflow/models/tree/master/research/audioset) and the audio-guided visual attention model was implemented highly based on [adaptive attention](https://github.com/jiasenlu/AdaptiveAttention). We thank the authors for sharing their codes. If you use our codes, please also cite their nice works.
 



