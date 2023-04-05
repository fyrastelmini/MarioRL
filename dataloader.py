import pandas as pd
import cv2
from tqdm import tqdm
import torch



def make_dataset(df):
    #get the images and resize them to (240, 256)
    def make_element(item,verbose=True):
        if verbose==True: pbar.update(1)
        return(cv2.imread(item))
    print("loading dataset frames")
    paths = df["paths"]#[0:2000]
    pbar=tqdm(total=len(paths))
    frames = [make_element(i) for i in paths]


    #get the labels
    labels = df[["A button","B button", "Start button","UP button","DOWN button","RIGHT button","LEFT button"]]#[0:2000]
    return frames,labels
def preprocess_frame(frame):
    # Resize the frame to 84x84
    frame = cv2.resize(frame, (84, 84))
    # Convert the frame to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Scale the pixel values to [0, 1]
    frame = frame.astype('float32') / 255.0
    # Return the preprocessed frame as a PyTorch tensor
    return torch.from_numpy(frame).unsqueeze(0)

def preprocess_data(frames, labels):
    preprocessed_frames = []
    for frame in frames:
        preprocessed_frames.append(preprocess_frame(frame))
    preprocessed_frames = torch.cat(preprocessed_frames, dim=0)
    # Convert the labels to a PyTorch tensor with one_hot encoding
    labels_one_hot = torch.tensor(labels.values, dtype=torch.float32)
    print(labels_one_hot)
    print(labels_one_hot.shape)
    print(preprocessed_frames.shape)
    return preprocessed_frames, labels_one_hot
