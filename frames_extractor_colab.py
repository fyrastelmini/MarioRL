import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx import all as fx
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image, ImageTk
import pytube
import os
from tqdm import tqdm
import argparse
import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument('--youtube_link', type=str, required=True, help='Link of the youtube demo video')
parser.add_argument('--start', type=int, default=0, required=False, help='Start of the video')
parser.add_argument('--end', type=int, default=60, required=False, help='End of the video')
args = parser.parse_args()


start=args.start
end=args.end

filename = 'data.mp4'  # replace with the actual file path and name
output_dir= "frames/"

if not(os.path.exists(filename)):
    # specify the URL of the YouTube video to download
    url = args.youtube_link

    # create a YouTube object from the URL
    yt = pytube.YouTube(url)

    # get the highest resolution stream and download the video
    stream = yt.streams.get_highest_resolution()
    stream.download(filename=filename)

    print('Video downloaded successfully.')
    
# set input and output file paths
input_file = "data.mp4"
#output_file = "data_cropped.mp4"
global_coords=[]
things_to_do=["gameplay screen","A button","B button", "Start button", "Select button","UP button","DOWN button","RIGHT button","LEFT button"]

x, y, w, h= [487,90,705,556]
global_coords.append((x, y, w, h))
x, y, w, h= [156,93,10,11]
global_coords.append((x, y, w, h))
x, y, w, h= [140,92,10,10]
global_coords.append((x, y, w, h))
x, y, w, h= [118,95,11,6]
global_coords.append((x, y, w, h))
x, y, w, h= [103,95,9,5]
global_coords.append((x, y, w, h))
x, y, w, h= [76,81,9,9]
global_coords.append((x, y, w, h))
x, y, w, h= [77,97,7,6]
global_coords.append((x, y, w, h))
x, y, w, h= [85,88,7,8]
global_coords.append((x, y, w, h))
x, y, w, h= [69,87,6,9]
global_coords.append((x, y, w, h))
for i,thing in zip(range(len(global_coords),things_to_do)):
    x, y, w, h = global_coords[i]
    print(thing," coordinates are ",x, y, w, h)







"""

# Define the coordinates of the region to crop 
x, y, w, h = 458, 0, 825, 720
"""
x, y, w, h = global_coords[0]
clip = VideoFileClip(input_file, audio=False)
cropped_clip = fx.crop(clip.subclip(start, end), x1=x, y1=y, x2=x+w, y2=y+h)
pbar=tqdm(total=int(cropped_clip.fps*cropped_clip.duration))
paths=[]
for i, frame in enumerate(cropped_clip.iter_frames()):
    image = Image.fromarray(frame)
    image = image.resize((256, 240), resample=Image.BOX)
    image.save(os.path.join(output_dir, f'frame_{i}.png'))
    paths.append(os.path.join(output_dir, f'frame_{i}.png'))
    pbar.update(1)
pbar.close()
print('Cropped video saved successfully.')

def detect_button(img):
    img=np.array(img)
    # convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # define lower and upper bounds of blue color with some tolerance
    lower_blue = np.array([90, 150, 150])
    upper_blue = np.array([110, 255, 255])

    # create a mask that identifies blue pixels in the image
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # count the number of blue pixels in the mask
    num_blue_pixels = np.count_nonzero(mask)
    # count the number of pixels in the image
    num_pixels = mask.shape[0] * mask.shape[1]
    """
    # plot the image, mask, and masked image
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    # plot the original image
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')

    # plot the mask
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask')

    # plot the masked image
    ax[2].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Masked Image')

    # set common title for the subplots
    fig.suptitle('Blue Object Detection', fontsize=16)

    plt.show()
"""
    # if there are enough blue pixels, the button is considered pressed
    if num_blue_pixels >= int(0.2 * num_pixels):
        #print("button pressed")
        return 1
    else:
        return 0

buttons=[]
for i in range(1,len(global_coords)):
    x, y, w, h = global_coords[i]
    print("extracting the "+things_to_do[i]+"\n")
    cropped_clip = fx.crop(clip.subclip(start, end), x1=x, y1=y, x2=x+w, y2=y+h)
    pbar=tqdm(total=int(cropped_clip.fps*cropped_clip.duration))
    button=[]
    for i, frame in enumerate(cropped_clip.iter_frames()):
        image = Image.fromarray(frame)
        image = image.resize((256, 240), resample=Image.BOX)
        button.append(detect_button(image))
        pbar.update(1)
    pbar.close()
    buttons.append(button)
dictionary = {things_to_do[i]: buttons[i-1] for i in range(1, len(things_to_do))}
df=pd.DataFrame(dictionary,columns=things_to_do[1:])
df["paths"]=paths
df.to_csv("data.csv",index=False)