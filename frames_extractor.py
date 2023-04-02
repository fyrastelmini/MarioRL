import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx import all as fx
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image
import pytube
import os
from tqdm import tqdm

start=17
end=60

filename = 'data.mp4'  # replace with the actual file path and name
output_dir= "frames/"
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
x, y, w, h = 458, 0, 825, 720

clip = VideoFileClip(input_file, audio=False)
cropped_clip = fx.crop(clip.subclip(start, end), x1=x, y1=y, x2=x+w, y2=y+h)
pbar=tqdm(total=cropped_clip.fps*cropped_clip.duration)
for i, frame in enumerate(cropped_clip.iter_frames()):
    image = Image.fromarray(frame)
    image = image.resize((256, 240), resample=Image.BOX)
    image.save(os.path.join(output_dir, f'frame_{i}.png'))
    pbar.update(1)
    
print('Cropped video saved successfully.')