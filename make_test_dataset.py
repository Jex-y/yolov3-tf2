import numpy as np
import os
from PIL import Image
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_integer("num_examples",1000,"Number of examples")
flags.DEFINE_integer("size",480,"Size of images to be made")
flags.DEFINE_string("output_data","raw_test_data","path to output data")
flags.DEFINE_integer("seed",0,"Random seed")

def main(_argv):
    if not os.path.isdir(FLAGS.output_data):
        os.mkdir(FLAGS.output_data)
    os.chdir(FLAGS.output_data)
    file_template = "image{}.jpg\t{}\t{}\t{}\t{}\t{}\n"
    name_template = ("image{}.jpg", "example{}.txt")
    np.random.seed(FLAGS.seed)
    for i in range(FLAGS.num_examples):
        image, ((xmin, xmax), (ymin, ymax)) = make_image((FLAGS.size,FLAGS.size))
        Image.fromarray(image,"RGB").save(name_template[0].format(i))
        with open(name_template[1].format(i),"w") as f:
            f.write(file_template.format(
                i,
                xmin,
                ymin,
                xmax - xmin,
                ymax - ymin,
                "test_data"))

        

def make_image(size):
    image = np.zeros((size[1],size[0],3),dtype=np.uint8)
    bbox = np.sort(np.random.rand(2,2)) * np.array([[size[0]],[size[1]]])
    bbox = bbox.astype(np.int32)
    image[
        bbox[1][0]:bbox[1][1],
        bbox[0][0]:bbox[0][1]] = np.array([255,255,255])
    return image, bbox


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
