import pandas as pd
import cv2
import matplotlib.pyplot as plt
import time
import matplotlib
df=pd.read_csv("data.csv")

for i in df["paths"]:
    img=cv2.imread(i)
    #buttons = list of column names that have 1 in them for the line that i is in
    buttons = df.columns[df.loc[df["paths"]==i].values[0]==1].tolist()
    if len(buttons)>0:
    #plot the image
        plt.imshow(img)
        plt.title(buttons)
        plt.show()
        #close plot after 0.1 seconds
        time.sleep(0.1)
        plt.close(fig="all")
