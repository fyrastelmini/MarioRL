import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx import all as fx
import pytube
import os

filename = 'data.mp4'  # replace with the actual file path and name

if not(os.path.exists(filename)):
    # specify the URL of the YouTube video to download
    url = 'https://www.youtube.com/watch?v=wx_mJUWqHvs'

    # create a YouTube object from the URL
    yt = pytube.YouTube(url)

    # get the highest resolution stream and download the video
    stream = yt.streams.get_highest_resolution()
    stream.download(filename=filename)

    print('Video downloaded successfully.')
    
# set input and output file paths
input_file = "data.mp4"
output_file = "data_cropped.mp4"
# Define the coordinates of the region to crop
x, y, w, h = 455, 0, 825, 720

clip = VideoFileClip(input_file)
cropped_clip = fx.crop(clip.subclip(0, clip.duration), x1=x, y1=y, x2=x+w, y2=y+h)
cropped_clip.write_videofile(output_file)
    
print('Cropped video saved successfully.')