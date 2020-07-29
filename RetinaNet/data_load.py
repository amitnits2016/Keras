import os
import sys
import math
import glob
import argparse
import xml.etree.ElementTree as ET

# file locations
# input - CHANGE THE DIR HERE TO TRAIN ON NEW DATA
#train_data = 'D:/P&G/RETINANET/keras-retinanet-master/Data/Train/'
#test_data = 'D:/P&G/RETINANET/keras-retinanet-master/Data/Test/'

# output
# train_annotations = 'D:/P&G/RETINANET/keras-retinanet-master/annotations.csv'
# val_annotations = 'D:/P&G/RETINANET/keras-retinanet-master/val_annotations.csv'
# label_file = 'D:/P&G/RETINANET/keras-retinanet-master/classes.csv'

train_annotations = './annotations.csv'
val_annotations = './val_annotations.csv'
label_file = './classes.csv'


# CONVERT the XML annotations to CSV format
def convert_annotation(train_data, test_data, image_id, filename, classes, train=False):
    if train:
        in_file = open(train_data + '%s.xml' % image_id)
    else:
        in_file = open(test_data + '%s.xml' % image_id)
    out_file = open(filename, 'a')
    tree = ET.parse(in_file)
    root = tree.getroot()

    if root.iter('object') is not None:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)

            xmlbox = obj.find('bndbox')
            x1 = math.ceil(float(xmlbox.find('xmin').text))
            y1 = math.ceil(float(xmlbox.find('ymin').text))
            x2 = math.ceil(float(xmlbox.find('xmax').text))
            y2 = math.ceil(float(xmlbox.find('ymax').text))
            if x1 == x2 or y1 == y2:
                continue
            if train:
                out_file.write(
                    f'{train_data + image_id}.jpg,{x1},{y1},{x2},{y2},{cls}\n')
            else:
                out_file.write(
                    f'{test_data + image_id}.jpg,{x1},{y1},{x2},{y2},{cls}\n')
    else:
        if train:
            out_file.write(f'{train_data + image_id}.jpg,,,,,\n')
        else:
            out_file.write(f'{test_data + image_id}.jpg,,,,,\n')


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    # subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    # subparsers.required = True

    # group = parser.add_mutually_exclusive_group()
    parser.add_argument('--train', help='Input dir with train images', default=' ', dest='train')
    parser.add_argument('--test',  help='Input dir for test images', default='', dest='test')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    train_data = args.train
    test_data = args.test

    # filter out the train test jpg files
    train_image = glob.glob(train_data + '/*.jpg')

    test_image = glob.glob(test_data + '/*.jpg')

    # define the classes for the model
    labels = ['Mistracking', 'Trim', 'Contamination', 'Tear', 'Wrinkle/Fold']

    # create new annotation files for  Train/validation
    open(train_annotations, 'w')
    open(val_annotations, 'w')

    train_ids = [os.path.basename(i[:-4]) for i in train_image]
    for image_id in train_ids:
        convert_annotation(train_data, test_data, image_id, train_annotations, labels, train=True)

    val_ids = [os.path.basename(i[:-4]) for i in test_image]

    for image_id in val_ids:
        convert_annotation(train_data, test_data, image_id, val_annotations, labels)

    # creating classes file listing all classes
    with open(label_file, 'w') as f:
        for i, line in enumerate(labels):
            f.write('{},{}\n'.format(line, i))

    print('DONE')


if __name__ == '__main__':
    main()

