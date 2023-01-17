import os
import torch

SPATH = ".data"
DPATH = ".dataset"

def organizeFiles(source_path, dest_path):
    unsorted_clips = [c for c in os.listdir(source_path) if "lab" not in c]
    unsorted_labels = [l for l in os.listdir(source_path) if "lab" in l]
    clips = sorted(unsorted_clips, key=lambda x: int(x.split('.')[0]))
    labels =  sorted(unsorted_labels, key=lambda x: x.split('.')[0])
    idx = 0

    for clip in clips:
        tmp = torch.load(clip, allow_pickle=True)['arr_0']
        for c in tmp.shape[0]:
            torch.save(c, f"{dest_path}/{idx}_c.pt")
            idx += 1
    
    label_list = []
    for label in labels:
        label_list.append(torch.load(label, allow_pickle=True)['arr_0'])
    
    label_list = torch.tensor(label_list)
    torch.save(label_list, f"{dest_path}/labels.pt")

if __name__ == "__main__":
    organizeFiles(SPATH, DPATH)