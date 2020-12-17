# Customer Confusion Detection Using LSTM
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
  <li><a href="#workflow-of-the-project">Workflow of the Project</a></li>
  <li><a href="#workflow-of-the-project">Data Preprocessing</a>
    <ul><li><a href="#workflow-of-the-project">Annotation Tools</a></li></ul>
    <ul><li><a href="#workflow-of-the-project">Features Aggregation</a></li></ul>
    <ul><li><a href="#workflow-of-the-project">Features Selection</a></li></ul>
    <ul><li><a href="#workflow-of-the-project">Sliding Windows</a></li></ul>
</li>
 <li><a href="#models">Models</a>
    <ul><li><a href="#basic-model">Basic Model</a></li></ul>
    <ul><li><a href="#vanilla-neural-network">Vanilla neural network</a></li></ul>
    <ul><li><a href="#LSTM">LSTM</a></li></ul>
    <ul><li><a href="#multi-input-LSTM">Multi-input LSTM</a></li></ul>
    <ul><li><a href="#bidirectional-LSTM">Bidirectional LSTM</a></li></ul>
<li><a href="#team-members">Team Members</a></li>
  </ol>
</details>

## Workflow of the Project
1. Data Proprocessing
    * We are using Annotation tools that was developed by previous people. [GitHub](https://github.com/leomorpho/confusion_detection)
    * data_aggregator.py will create the dataset in floder "./processed_data"
        * All the 60 features stored in "./processed_data/all/*json"
        * Hat orientation features stored in "./processed_data/hat/*json"
        * Openpose coordinates stored in "./processed_data/openpose/*json"
    * Feature selection will reduce the features to 34 and store the data in "./processed_data/feature_selected.json"
        * ` python3 feature_selection.py `
    * Lastly, create the video sequences and sotre them as npy file
        * `python3 sliding_windows.py`
![Data distribution](/img/slidingwindow.jpg "The distribution of dataset")

2. Models 
    * We have implemented more than 5 different models for comparsion. 

## Models

### Basic model
* It will output the accuracy and 5-flod cv of three basic models, SVM, KNN, Randon Forest
* It will plot the data distribution at the end
* `python basic_model.py`
![Data distribution](/img/data_dist.png "The distribution of dataset")

### Vanilla neural network
* `python vanilla_neural_net.py`
* It will initialize a basic deeply connected neural network
* It will train and validate the model 
* It will plot the accuracy/loss of training and validation dataset at the end

### LSTM
* `python model.py`
* It will initialize a model with two layer of LSTM and a output layer
* It will train and validate the model 
* It will plot the accuracy/loss of training and validation dataset at the end

### Multi-input LSTM
* `python multi_LSTM.py`
* It will initialize a combined LSTM model that takes hat orientation and OpenPose coordinates as separate input
* It will train and validate the model 
* It will plot the accuracy/loss of training and validation dataset at the end

### Bidirectional LSTM
* `python bidirectional.py` will show the result plot
* which_lstms(frame_sequences,labels) will show you which lstm is better
* which_lstm_mode(frame_sequences,labels) will show you which lstm merge modes is better
![Bidirectional LSTM result](/img/bidirectional_lstm.png "Bidirectional LSTM result")
### Team Members
* Xu Zhicheng - maxx@sfu.ca
* Li Rong - rong_li_2@sfu.ca
* Xuecong Tan - xuecongt@sfu.ca

