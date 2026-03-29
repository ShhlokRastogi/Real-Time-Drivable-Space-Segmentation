import os
mask_dir = 'c:/drivableseg/merged_road_dataset/train/masks'
masks = os.listdir(mask_dir)
for m in masks:
    if '1535657110154799' in m:
        print("MATCH:", m)
