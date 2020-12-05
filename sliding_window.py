import json
import glob
import numpy as np

WINDOW_SIZE = 5
SLIDE = 1

IS_ALL = True
IS_HAT = False
IS_OP = False
# this implementation is for baseline LSTM model training
def main():
    data = []
    # obtain path for training data
    if IS_ALL:
        train_path = glob.glob("processed_data/all/*.json")
    elif IS_HAT:
        train_path = glob.glob("processed_data/hat/*.json")
    elif IS_OP:
        train_path = glob.glob("processed_data/openpose/*.json")

    # concatenating all video frames to one giant dataset 
    for path in train_path:
        with open(path) as f:
            raw_data = json.loads(f.read())
        data.append(raw_data)

    num_features = len(data[0][0]) - 1 # do not include label as a feature

    sequential_data = []
    sequential_labels = []
    
    for lst in data:
        window_left = 0
        window_right = WINDOW_SIZE - 1
        
        while window_right < len(lst):
            window = []
            for i in range(window_left, window_right + 1):
                window.append(lst[i][1:])
                if i == window_right:
                    sequential_labels.append(lst[i][0])
            sequential_data.append(window)
            window_left += SLIDE
            window_right += SLIDE

    data_np = np.array(sequential_data)
    labels_np = np.array(sequential_labels)

    np.save('LSTM_input.npy', data_np)
    np.save('LSTM_labels.npy', labels_np)


if __name__ == '__main__':
    main()