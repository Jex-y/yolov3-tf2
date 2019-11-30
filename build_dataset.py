import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io
import hashlib

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    
def _float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def make_example(file,data_path="data"):
    with open(data_path + "\\" + file,"r") as f:
            data = f.read().split("\n")

    xmin, xmax, ymin, ymax, text, label, diff, trunc, view = [], [], [], [], [], [], [], [], []
    size = None
    for meta in data:
        if meta != "":
            meta = meta.split("\t")
            image_path = meta[0]
            if size == None:
                image_path = os.path.abspath(data_path + "\\" + image_path)
                with tf.io.gfile.GFile(image_path,"rb") as fid:
                    encoded_jpg = fid.read()
                key = hashlib.sha256(encoded_jpg).hexdigest()  
                encoded_jpg_io = io.BytesIO(encoded_jpg)
                img =  Image.open(encoded_jpg_io)
                size = img.size
                img.close()

            x = [float(meta[1]) / size[0],
                 float(meta[3]) / size[0]]

            y = [float(meta[2]) / size[1],
                 float(meta[4]) / size[1]]
           
            xmin.append(min(x))
            xmax.append(max(x))
            ymin.append(min(y))
            ymax.append(max(y))

            text.append("Number Plate".encode("utf8"))
            label.append(0)
            diff.append(False) # Is object difficult to regocnise e.g. hidden or obstructed
            trunc.append(False) # It the object only partial e.g. person from waist up
            view.append("back".encode("utf8")) # View of the object e.g. behind or infront 

            plate = meta[5]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
            "image/height"              : _int64_feature(size[1]),
            "image/width"               : _int64_feature(size[0]),
            "image/filename"            : _bytes_feature(image_path.encode("utf8")),
            "image/source_id"           : _bytes_feature(image_path.encode("utf8")),
            "image/key/sha256"          : _bytes_feature(key.encode("utf8")),
            "image/encoded"             : _bytes_feature(encoded_jpg),
            "image/format"              : _bytes_feature("jpg".encode("utf8")),
            "image/object/bbox/xmin"    : _float_list_feature([x/size[0] for x in xmin]),
            "image/object/bbox/xmax"    : _float_list_feature([x/size[0] for x in xmax]),
            "image/object/bbox/ymin"    : _float_list_feature([y/size[1] for y in ymin]),
            "image/object/bbox/ymax"    : _float_list_feature([y/size[1] for y in ymax]),
            "image/object/class/text"   : _bytes_list_feature(text),
            "image/object/class/label"  : _int64_list_feature(label),
            "image/object/difficult"    : _int64_list_feature(diff),
            "image/object/truncated"    : _int64_list_feature(trunc),
            "image/object/view"         : _bytes_list_feature(view)
        }))

    return tf_example


def build(output_path,data_path="data",val_split=0.2,seed=0):
    np.random.seed(seed)
    files = [ x for x in os.listdir(data_path) if x.split(".")[-1] == "txt"]
    val = np.random.rand(len(files))
    indexes_all = [np.nonzero(val>=val_split)[0],np.nonzero(val<val_split)[0]]
    print("Number of train samples: {}\nNumber of validation samples {}\n".format(indexes_all[0].shape[0],indexes_all[1].shape[0]))
    count = 0
    size = 50
    for indexes in indexes_all:
        num = indexes.shape[0]
        done = 0
        writer = tf.io.TFRecordWriter(output_path.format("train" if count == 0 else "val"))
        for index in indexes:
            tf_example = make_example(files[index],data_path=data_path)
            writer.write(tf_example.SerializeToString())
            done += 1
            ratio = done/num
            print("\r{} dataset {:.2f}% Complete\t|{}{}{}|\t".format("training" if count == 0 else "validation",ratio*100,"\u2588"*int(ratio*size),">" if ratio < 1 else "\u2588"," "*int((1-ratio)*size)),end="")
        writer.close()
        print("")
        count += 1

if __name__ == "__main__":
    build(".\\data\\{}.tfrecord",data_path="raw_data",val_split=0.2)