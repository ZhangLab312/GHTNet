# GHTNet
GHTNet is a deep learning-based framework that can be used to accurately identify TF-DNA binding specificity from the human and mouse genome. The performance of GHTNet was evaluated on the 86 human ChIP-Seq datasets and 9 mouse ChIP-Seq datasets, which is downloaded from ENCODE (https://www.encodeproject.org/matrix/?type=Experiment&status=released&biosample_ontology.classification=tissue). Using GHTNet, we performed many analyses about tissue-specific TF-DNA binding, following figure show an overview. 
![image](https://github.com/ZhangLab312/GHTNet/blob/main/overview.png)

# Prerequisites and Dependencies  
* Pytorch 1.10
* CUDA 11.1
* Python 3.7
* Numpy 1.21.6
* Pandas 1.3.4
* Tqdm 4.62.3
* Sklearn

# Data preparation
In this study, we utilized ChIP-Seq datasets from ENCODE (https://www.encodeproject.org/matrix/?type=Experiment&status=released&biosample_ontology.classification=tissue). All these datasets were processed by using the python package "deepTools" (v1.0.3, (Ramrez et al., 2016)) and the R package "GKMSVM" (v0.81.1, (Ghandi et al., 2016)). In addition, the R package "DNAshapeR"  (v1.9.5, (Li et al., 2017)) was used to generate DNA shape feature and "phastCons100way" (v3.15, (Siepel et al., 2005)) was used to generate conservation score. Finally, five type of data were obtained, including DNA sequence, DNA shape, histone modification, DNase, and conservation score.

# Usage
GHTNet was a framework to predict TF-DNA specificity. Please use the Train.py to train the model. Firstly, read_data.py was used to encode five type of feature. For DNA sequence, word2vec strategy was used to obtain distributed representation. Then we leveraged the encoder part of Transformer architecture. Finally, we used CNN and MLP to predict TF-DNA binding. Early stop strategy was used to get final model. After completing the training, The attention map for each attention head and
the essential gene regions that influence model decisions were visualized. And a similar strategy in DeepBind was used to obtain motifs.<br>

Beforing using please checking the file folder path.

# Example
Due to file size limitation, we only uploaded CTCF data on middle frontal area 46 tissue (including 5 type of data: DNA sequence, DNA shape, histone modification, DNase, and conservation score) as an example. Users can build data as the format of the example.

# Train GHTNet
‘python train.py’ for linux or use pycharm to run the train.py file directly. Hyperparameters can be set in the python file.

# Citation
If you use GHTNet in your research, please cite the following paper: <br>

"Uncovering the relationship between tissue-specific TF-DNA binding and chromatin features through a Transformer-based model", <br>


