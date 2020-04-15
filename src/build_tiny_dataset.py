import csv
import os
import shutil
from glob import glob
from classes import mapper


TRAIN_CSV_PATH = os.path.abspath('../datasets/data/nyu2_train.csv')
TRAIN_PATH = '../datasets/data/train/'


def makedir(path):
    if os.path.exists(path):
        return
    os.mkdir(path)


def read_csv(csv_file_path):
    with open(csv_file_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        img_dm_pairs = [(row[0], row[1]) for row in csv_reader if len(row) > 0]
        return img_dm_pairs


def write_csv(csv_file_path):
    images = glob("../datasets/data/nyu2_train/*/*.jpg")
    images.sort(key=lambda e: e.split('/')[-1].split('.')[0])

    depth_maps = glob("../datasets/data/nyu2_train/*/*.png")
    depth_maps.sort(key=lambda e: e.split('/')[-1].split('.')[0])

    rows_list = []
    for img, dm in zip(images, depth_maps):
        img_path = img.split('/')
        img_path = '{}/{}/{}/{}'.format(img_path[2],
                                        img_path[3], img_path[4], img_path[5])

        dm_path = dm.split('/')
        dm_path = '{}/{}/{}/{}'.format(dm_path[2],
                                       dm_path[3], dm_path[4], dm_path[5])

        rows_list.append([img_path, dm_path])

    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(rows_list)


def make_dataset(csv_pairs, mapping_dict):
    for img, dm in csv_pairs:
        img_path_split = img.split('/')
        dm_path_split = dm.split('/')
        img_class = img_path_split[2]
        img_name = img_path_split[2] + "_" + img_path_split[3]
        dm_name = dm_path_split[2] + "_" + dm_path_split[3]

        class_folder_name = mapping_dict[img_class]
        class_folder_path = os.path.abspath('../datasets/data/train/' + class_folder_name)
        makedir(class_folder_path)

        img_src_path = os.path.abspath('../datasets/' + img)
        dm_src_path = os.path.abspath('../datasets/' + dm)

        img_dst_path = os.path.abspath(class_folder_path + "/" + img_name)
        dm_dst_path = os.path.abspath(class_folder_path + "/" + dm_name)

        shutil.copy2(img_src_path, img_dst_path)
        shutil.copy2(dm_src_path, dm_dst_path)


if __name__ == "__main__":
    write_csv('../datasets/data/nyu2_train.csv')

    # z = [x[0].split('/')[-1] for x in os.walk('../datasets/data/nyu2_train/')]
    # mapping_dict = {}

    # for i in z:
    #     mapper(i, mapping_dict)
    
    # csv_pairs = read_csv(TRAIN_CSV_PATH)
    # make_dataset(csv_pairs, mapping_dict)