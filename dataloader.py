import pandas as pd
import cv2
from tqdm import tqdm




def make_dataset(df):
    #get the images and resize them to (240, 256)
    def make_element(item,verbose=True):
        if verbose==True: pbar.update(1)
        return(cv2.resize(cv2.imread(item), (256, 240)))

    paths = df["paths"]#[0:2000]
    pbar=tqdm(total=len(paths))
    frames = [make_element(i) for i in paths]


    #get the labels
    labels = df[["A button","B button", "Start button","UP button","DOWN button","RIGHT button","LEFT button"]]#[0:2000]
    return frames,labels