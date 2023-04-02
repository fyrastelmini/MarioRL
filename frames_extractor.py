import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm
import pytube

# specify the URL of the YouTube video to download
url = 'https://www.youtube.com/watch?v=wx_mJUWqHvs'

# create a YouTube object from the URL
yt = pytube.YouTube(url)

# get the highest resolution stream and download the video
stream = yt.streams.get_highest_resolution()
stream.download(filename='data.mp4')

print('Video downloaded successfully.')

pbar=tqdm(total=8)
# set input and output file paths
input_file = "data.mp4"
output_file = "data_cropped.mp4"
# Define the coordinates of the region to crop
x, y, w, h = 455, 0, 825, 720

clip = VideoFileClip(input_file)
pbar.update(1)
# define cropping function
def crop_frame(frame):
    return frame[y:y+h, x:x+w]

# apply cropping function to each frame in the video clip
cropped_frames = [crop_frame(frame) for frame in clip.iter_frames()]
pbar.update(1)
# create new video clip object with cropped frames
cropped_clip = VideoFileClip(None, fps_source=30)
pbar.update(1)
cropped_clip.duration = clip.duration
pbar.update(1)
cropped_clip.fps = clip.fps
pbar.update(1)
cropped_clip.size = (w, h)
pbar.update(1)
cropped_clip.writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
pbar.update(1)

for frame in cropped_frames:
    cropped_clip.write_frame(frame)
pbar.update(1)
# close video writer and clip objects
cropped_clip.writer.release()
clip.reader.close()
clip.audio.reader.close_proc()