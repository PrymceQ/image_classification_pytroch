import os
import cv2
import argparse
import numpy as np

from glob import glob
from tqdm import tqdm

def count_jpg_files(folder_path):
    jpg_count_per_folder = {}

    for root, dirs, files in os.walk(folder_path):
        if root == folder_path:
            continue
        jpg_count = len([file for file in files if file.lower().endswith('.jpg')])
        folder_name = os.path.basename(root)
        jpg_count_per_folder[folder_name] = jpg_count

    return jpg_count_per_folder

def calculate_percentage_distribution(jpg_counts):
    total_jpgs = sum(jpg_counts.values())
    percentage_distribution = {folder: (count / total_jpgs) * 100 for folder, count in jpg_counts.items()}
    return percentage_distribution

def cal_meanstd(folder_path):
    img_paths = glob(f'{folder_path}/*/*.jpg')
    m_list, s_list = [], []
    for img_path in tqdm(img_paths):
        if not img_path.endswith('jpg'):
            continue
        # BGR
        img = cv2.imread(img_path)
        # RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))

    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)

    return m[0], s[0]

if __name__=='__main__':
    parser = argparse.ArgumentParser('Classification inference code!!')
    parser.add_argument('--folder', type=str, default='dataset/test', help='image main folder path')
    opt = parser.parse_args()

    # 统计分布
    jpg_counts = count_jpg_files(opt.folder)
    for folder, count in jpg_counts.items():
        print(f"Folder '{folder}' contains {count} JPG images.")
    percentage_distribution = calculate_percentage_distribution(jpg_counts)
    for folder, percentage in percentage_distribution.items():
        print(f"Folder '{folder}' contains {percentage:.2f}% of total JPG images.")

    # 计算mean和std
    m, s = cal_meanstd(opt.folder)
    print('RGB\nMean: {}\nStd:  {}'.format(m, s))

