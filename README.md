Audio-Visual Event Localization in Unconstrained Videos (To appear in ECCV 2018) 

[Project](https://sites.google.com/view/audiovisualresearch) [ArXiv](https://arxiv.org/abs/1803.08842)

### AVE Dataset & Features

AVE dataset can be downloaded from https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK.

[Audio feature](https://drive.google.com/file/d/1F6p4BAOY-i0fDXUOhG7xHuw_fnO5exBS/view?usp=sharing) and [visual feature](https://drive.google.com/file/d/1hQwbhutA3fQturduRnHMyfRqdrRHgmC9/view?usp=sharing) (7.7GB) are also released. Please put videos of AVE dataset into /data/AVE folder and features into /data folder before running the code. 

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



### Citation

If you find this work useful, please consider citing it.

<pre><code>@InProceedings{tian2018ave,
  author={Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chenliang Xu},
  title={Audio-Visual Event Localization in Unconstrained Videos},
  booktitle = {ECCV},
  year = {2018}
}
</code></pre>
 
 ### Acknowledgements
 
Audio features are extracted using [vggish](https://github.com/tensorflow/models/tree/master/research/audioset) and the audio-guided visual attention model was implemented highly based on [adaptive attention](https://github.com/jiasenlu/AdaptiveAttention). We thank the authors for sharing their codes. If you use our codes, please also cite their nice works.
 



