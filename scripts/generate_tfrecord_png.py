"""
Usage:
  python generate_tfrecord.py --data_dir=PATH_TO_Cold_brew --output_path=PATH_TO_output.record --label_map_path=PATH_TO_label_map.pbtxt
"""

import os
import glob
import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import argparse

def create_tf_example(example, annotation_dir, image_dir, label_map_dict):
    # Find the image file with supported extensions
    for ext in ['.jpg', '.jpeg', '.png']:
        img_path = os.path.join(image_dir, example + ext)
        if os.path.exists(img_path):
            image_format = ext[1:].encode('utf8')  # e.g. "jpg"
            break
    else:
        raise FileNotFoundError(f"No image found for {example} with .jpg/.png")

    # Read the image binary
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_image_data = fid.read()

    # Parse the corresponding annotation XML file
    annotation_path = os.path.join(annotation_dir, example + '.xml')
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Read image size
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    # Lists for bounding box coordinates and labels
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # Extract object annotations from XML
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in label_map_dict:
            continue  # Skip if not in label map
        classes_text.append(name.encode('utf8'))
        classes.append(label_map_dict[name])

        bndbox = obj.find('bndbox')
        xmins.append(float(bndbox.find('xmin').text) / width)
        xmaxs.append(float(bndbox.find('xmax').text) / width)
        ymins.append(float(bndbox.find('ymin').text) / height)
        ymaxs.append(float(bndbox.find('ymax').text) / height)

    # Build the TF Example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(img_path.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(img_path.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Root directory to dataset (Cold_brew)')
    parser.add_argument('--output_path', required=True, help='Path to output TFRecord')
    parser.add_argument('--label_map_path', required=True, help='Path to label_map.pbtxt')
    args = parser.parse_args()

    # Load label map and convert to dictionary
    label_map_dict = label_map_util.get_label_map_dict(label_map_util.load_labelmap(args.label_map_path))

    # Construct sub-directory paths
    annotation_dir = os.path.join(args.data_dir, 'Annotations')
    image_dir = os.path.join(args.data_dir, 'JPEGImages')
    split_file = os.path.join(args.data_dir, 'ImageSets', 'Main', 'default.txt')

    # Initialize TFRecord writer
    writer = tf.io.TFRecordWriter(args.output_path)

    # Read list of image IDs to include in TFRecord
    with open(split_file, 'r') as f:
        examples = [x.strip() for x in f.readlines()]

    # Process each example
    for example in examples:
        tf_example = create_tf_example(example, annotation_dir, image_dir, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('TFRecord created at:', args.output_path)

if __name__ == '__main__':
    main()