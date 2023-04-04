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
for thing in things_to_do:
    # create a Tkinter window and canvas to display the video frame
    root = tk.Tk()
    root.title("Select the "+thing+" area")
    # use OpenCV to capture frames from the video
    cap = cv2.VideoCapture(input_file)
    ret, frame = cap.read()

    # convert the OpenCV frame to a PIL image and display it on the canvas
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    photo = ImageTk.PhotoImage(img)
    #img.save("debug.png")

    canvas = tk.Canvas(root, width=photo.width(), height=photo.height(), borderwidth=0, highlightthickness=0)

    canvas.create_image(0, 0, anchor='nw', image=photo)
    canvas.pack(side='top', padx=0, pady=0)

    # add a cropping tool to the canvas
    crop_rect = canvas.create_rectangle(0, 0, 0, 0, outline='red')
    crop_coords = [0, 0, 0, 0]

    def start_crop(event):
        global crop_coords
        crop_coords[0], crop_coords[1] = event.x, event.y

    def update_crop(event):
        global crop_rect, crop_coords
        canvas.coords(crop_rect, crop_coords[0], crop_coords[1], event.x, event.y)
        crop_coords[2], crop_coords[3] = event.x - crop_coords[0], event.y - crop_coords[1]

    def end_crop(event):
        global x, y, w, h, crop_coords
        x, y, w, h = crop_coords
        root.destroy()

    # bind mouse events to the cropping tool
    canvas.bind('<ButtonPress-1>', start_crop)
    canvas.bind('<B1-Motion>', update_crop)
    canvas.bind('<ButtonRelease-1>', end_crop)

    # update the Tkinter window
    root.mainloop()
    global_coords.append((x, y, w, h))
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
    #received a cropped image containing a NES controller button and detects if the button is colord blue
    #returns 1 if there's blue, 0 otherwise
    img=img.convert('RGB')
    r,g,b=img.split()
    r=np.array(r)
    g=np.array(g)
    b=np.array(b)
    if np.sum(b)>np.sum(r) and np.sum(b)>np.sum(g):
        return 1
    else:
        return 0
buttons=[]
for i in range(1,len(global_coords)):
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
df.to_csv("data.csv")