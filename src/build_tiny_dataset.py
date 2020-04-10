import csv
import os
import shutil
from glob import glob
from classes import mapping_dict


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
    # images = glob("../datasets/data/nyu2_train/*/*.jpg")
    # images.sort(key=lambda e: int(e.split('/')[-1].split('.')[0]))

    # depth_maps = glob("../datasets/data/nyu2_train/*/*.png")
    # depth_maps.sort(key=lambda e: int(e.split('/')[-1].split('.')[0]))

    # rows_list = [[i[0], i[1]] for i in zip(images, depth_maps)]

    # with open('../datasets/data/new/nyu2_train.csv', 'w', newline='') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerows(rows_list)
    pass


def make_dataset(csv_pairs):
    for img, dm in csv_pairs:
        img_path_split = img.split('/')
        dm_path_split = dm.split('/')
        img_class = img_path_split[4]
        img_name = img_path_split[4] + "_" + img_path_split[5]
        dm_name = dm_path_split[4] + "_" + dm_path_split[5]

        class_folder_name = mapping_dict[img_class]
        class_folder_path = os.path.abspath(TRAIN_PATH + class_folder_name)
        makedir(class_folder_path)

        img_src_path = os.path.abspath(img)
        dm_src_path = os.path.abspath(dm)

        img_dst_path = os.path.abspath(class_folder_path + "/" + img_name)
        dm_dst_path = os.path.abspath(class_folder_path + "/" + dm_name)

        shutil.copy2(img_src_path, img_dst_path)
        shutil.copy2(dm_src_path, dm_dst_path)


if __name__ == "__main__":
    csv_pairs = read_csv(TRAIN_CSV_PATH)
    make_dataset(csv_pairs)
