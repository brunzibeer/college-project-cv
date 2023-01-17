import json
import torch
import os
import numpy as np

from sklearn import preprocessing as pp
from moviepy.video.io.VideoFileClip import VideoFileClip as vfc

VIDEO_PATH = "placeholder/path"
CLIP_PATH = "placeholder/path"
ENCODER = pp.LabelEncoder() # Label conversion: string -> int
DESIRED_SHAPE = (224, 398, 3) # Single frame shape

# Some videos hava a different resolution so I need to validate them
def validateClip(clip):
    if clip.shape[-3:] != DESIRED_SHAPE:
        return False
    
    return True

# Extracting slected frames from a video
def extractFrames(video, window_size, source_fps, fps, starting_frame):
    '''
    Args:
        - video: input video
        - window_size: time window of the action
        - source_fps: fps of the original video
        - fps: frame per second during sampling
        - starting_frame: middle frame of the action
    
    Return:
        - clip: torch tensor of shape [30, 224, 398, 3]
    '''
    offset = (window_size / 2) * source_fps
    sample_rate = source_fps // fps
    first_frame = starting_frame - offset
    sample_number = window_size * fps
    clip = list()

    for i in range(sample_number):
        print(f"Sampling the {i}th frame: {video[first_frame, :, :, :].shape}")
        clip.append(video[first_frame, :, :, :])
        first_frame = first_frame + sample_rate
    
    clip = torch.cat(clip, dim=0)
    return clip

def getClip(actions = ['Goal', 'Corner', 'Foul'], fps=3, window_size=10, th=10):
    '''
    Args:
        - actions: array of the actions I want to get
        - fps: frame per second of the extracted clip
        - window_size: time window of the action
        - th: threshold for the number of instances of a certain class
    '''
    idx = 0
    championships = os.listdir(VIDEO_PATH)
    extracted_labels = list()
    final_dict = {
                    'Goal': 0,
                    'Foul': 0,
                    'Corner': 0,
                    'Nothing': 0,
                }
    for c in championships:
        years = os.listdir(f"{VIDEO_PATH}/{c}")
        for y in years:
            games = os.listdir(f"{VIDEO_PATH}/{c}/{y}")
            for g in games:
                prev_len = len(extracted_labels)
                g_path = f"{VIDEO_PATH}/{c}/{y}/{g}"
                l_path = g_path + "/Labels-v2.json"
                # I need to keep track of the instances for every game not to make a very unbalanced dataset
                d = {
                    'Goal': 0,
                    'Foul': 0,
                    'Corner': 0,
                    'Nothing': 0,
                }
                with open(l_path) as labels:
                    labs = json.load(labels)['annotations']
                    
                    for action in labs:
                        if action['label'] in actions:
                            label = action['label']
                        else:
                            label = 'Nothing'
                        if d[label] < th:
                            half, time = action['gameTime'].split('-')
                            minutes, seconds = map(int, time.strip(' ').split(':'))
                            seconds = seconds + (minutes * 60)

                            video_path = f"{g_path}/{half.strip(' ')}.mkv"
                            if not os.path.isfile(video_path):
                                continue
                            try:
                                with vfc(video_path) as v:
                                    clip = v.subclip(seconds - window_size/2, seconds + window_size/2)
                                    frames = list()
                                    for frame in clip.iter_frames(fps=fps):
                                        frames.append(frame)
                            except Exception:
                                print("Video Error!")
                            
                            clip_tensor = torch.tensor(frames)
                            if not validateClip(clip_tensor):
                                continue

                            d[label] += 1
                            final_dict[label] += 1
                            torch.save(clip_tensor, f"{CLIP_PATH}/{idx}_clip.pt")
                            idx += 1
                            extracted_labels.append(label)
                    
                    
                    print(f"Got the following actions from game {g}: {d}\n For a total of {sum(d.values())} actions with respective {len(extracted_labels) - prev_len} labels.")


    game_labels = np.array(extracted_labels)
    game_labels = torch.as_tensor(ENCODER.fit_transform(game_labels))
    print("Saving labels as tensor...")
    torch.save(game_labels, f"{CLIP_PATH}/labels.pt")

    print(f"Extraction Done! Got {final_dict} for a total of {sum(final_dict.values())} actions and a label tensor of shape {game_labels.shape}.")            


if __name__ == "__main__":
    getClip()