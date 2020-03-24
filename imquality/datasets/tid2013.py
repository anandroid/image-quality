import os.path

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from . import CHECKSUMS_PATH


tfds.download.add_checksums_dir(CHECKSUMS_PATH)


CITATION = r"""
@article{ponomarenko2015image,
  title={Image database TID2013: Peculiarities, results and perspectives},
  author={Ponomarenko, Nikolay and Jin, Lina and Ieremeiev, Oleg and Lukin, Vladimir and Egiazarian, Karen and Astola, Jaakko and Vozel, Benoit and Chehdi, Kacem and Carli, Marco and Battisti, Federica and others},
  journal={Signal Processing: Image Communication},
  volume={30},
  pages={57--77},
  year={2015},
  publisher={Elsevier}
}
"""
DESCRIPTION = """
The TID2013 contains 25 reference images and 3000 distorted images 
(25 reference images x 24 types of distortions x 5 levels of distortions). 
Reference images are obtained by cropping from Kodak Lossless True Color Image Suite. 
All images are saved in database in Bitmap format without any compression. File names are 
organized in such a manner that they indicate a number of the reference image, 
then a number of distortion's type, and, finally, a number of distortion's level: "iXX_YY_Z.bmp".
"""
URLS = b'http://www.ponomarenko.info/tid2013.htm'
SUPERVISED_KEYS = ("distorted_image", "mos")

MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    You can download the images from
    https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM
    Please look at the source file (cbis_ddsm.py) to see the instructions
    on how to convert them into png (using dcmj2pnm).
    """


class Tid2013(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
     You can download the images from
     https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM
     Please look at the source file (cbis_ddsm.py) to see the instructions
     on how to convert them into png (using dcmj2pnm).
     """

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "distorted_image": tfds.features.Image(),
                "reference_image": tfds.features.Image(),
                "mos": tf.float32,
            }),
            supervised_keys=SUPERVISED_KEYS,
            homepage=URLS,
            citation=CITATION,
        )

    def _split_generators(self, manager):
        tid2013 = "https://download844.mediafire.com/hd0m3bg8ifpg/3yv173a68nuy53a/tid2013.zip"
        #manager.manual_dir()
        extracted_path="/home/anandkumar/tensorflow_datasets/downloads/manual/"
        images_path = os.path.join(extracted_path,"tid2013")
        print("extracted_path")
        print(extracted_path)
        print("images path")
        print(images_path)

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "images_path": images_path,
                    "labels": os.path.join(images_path, "mos.txt")
                },
            )
        ]

    def _generate_examples(self, images_path, labels):
        with tf.io.gfile.GFile(labels) as f:
            lines = f.readlines()


        print("lines")
        print(lines)

        print("images_path")
        print(images_path)


        for image_id, line in enumerate(lines[1:]):
            values = line.split(",")
            yield image_id, {
                "distorted_image": os.path.join(images_path, values[0]),
                "reference_image": os.path.join(images_path, values[1]),
                "mos": values[2],
            }
