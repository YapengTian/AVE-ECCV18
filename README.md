Audio-Visual Event Localization in Unconstrained Videos (To appear in ECCV 2018) (not ready)

### AVE Dataset & Features

AVE dataset can be downloaded from https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK.

[Audio feature](https://drive.google.com/file/d/1F6p4BAOY-i0fDXUOhG7xHuw_fnO5exBS/view?usp=sharing) and [visual feature](https://drive.google.com/file/d/1hQwbhutA3fQturduRnHMyfRqdrRHgmC9/view?usp=sharing) (7.7GB) are also released. Please put videos of AVE dataset into /data/AVE folder and features into /data folder before running the code.

### Requirements

Python-3.6, Pytorch-0.3.1, Keras, ffmpeg. 

### Visualize attention maps

Run: python attention_visualization.py to generate audio-guided visual attention maps. 

![image](https://github.com/YapengTian/AVE-ECCV18/blob/master/Figs/att_easy.jpg)

### Supervised audio-visual event localization

Testing: 

A+V-att model in the paper: python supervised_main.py --model_name AV_att

[DMRN](https://drive.google.com/file/d/1hQwbhutA3fQturduRnHMyfRqdrRHgmC9/view?usp=sharing) model in the paper:    python supervised_main.py --model_name DMRN  

Training:

python supervised_main.py --model_name AV_att --train



### Weakly-supervised audio-visual event localization

### Cross-modality localization


### Citation

If you find this work useful, please consider citing it.

 > @inproceedings{AVE2018, <br>
 >    title={Audio-Visual Event Localization in Unconstrained Videos},<br>
 >   author={Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chenliang Xu},<br>
 >    booktitle={ECCV},<br>
 >   year={2018}<br>
 > }
 
 
 ### Acknowledgements
 
Audio features are extracted using [vggish](https://github.com/tensorflow/models/tree/master/research/audioset) and the audio-guided visual attention model was implemented highly based on [adaptive attention](https://github.com/jiasenlu/AdaptiveAttention). We thank the authors for sharing their codes. If you use our codes, please also cite their nice works.
 



