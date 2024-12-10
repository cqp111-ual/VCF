'''Shared code among the image entropy codecs. "Uncompressed" IO
(i.e., the input of encode and the output of decode) uses PNG.

'''

import os
import io
from skimage import io as skimage_io # pip install scikit-image
from PIL import Image # pip install 
import numpy as np
import logging
import subprocess
import cv2 as cv # pip install opencv-python
import main
import urllib

from information_theory import distortion # pip install "information_theory @ git+https://github.com/vicente-gonzalez-ruiz/information_theory"

class CoDec:

    def __init__(self, args):
        self.args = args
        logging.debug(f"args = {self.args}")
        if args.subparser_name == "encode":
            self.encoding = True
        else:
            self.encoding = False
        logging.debug(f"self.encoding = {self.encoding}")
        self.input_bytes = 0
        self.output_bytes = 0

    def __del__(self):
        logging.info(f"Total {self.input_bytes} bytes read")
        logging.info(f"Total {self.output_bytes} bytes written")
        if self.encoding:
            BPP = (self.output_bytes*8)/(self.img_shape[0]*self.img_shape[1])
            logging.info(f"rate = {BPP} bits/pixel")
            with open(f"{self.args.output}_BPP.txt", 'w') as f:
                f.write(f"{BPP}")
        else:
            if __debug__:
                img = self.encode_read_fn("file:///tmp/original.png")
                y = self.encode_read_fn(self.args.output)
                RMSE = distortion.RMSE(img, y)
                logging.info(f"RMSE = {RMSE}")
                with open(f"{self.args.input}_BPP.txt", 'r') as f:
                    BPP = float(f.read())
                J = BPP + RMSE
                logging.info(f"J = R + D = {J}")

    def encode(self):
        img = self.encode_read()
        compressed_img = self.compress(img)
        self.encode_write(compressed_img)
        #logging.info(f"BPP = {BPP}")
        #return BPP

    def decode(self):
        compressed_img = self.decode_read()
        img = self.decompress(compressed_img)
        #compressed_img_diskimage = io.BytesIO(compressed_img)
        #img = np.load(compressed_img_diskimage)['a']
        #decompressed_data = zlib.decompress(compressed_img)
        #img = io.BytesIO(decompressed_data))
        self.decode_write(img)
        #logging.debug(f"output_bytes={self.output_bytes}, img.shape={img.shape}")
        #self.BPP = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        #return rate, 0
        #logging.info("RMSE = 0")

    def encode_read(self):
        '''Read the image specified in the class attribute <args.input>.'''
        img = self.encode_read_fn(self.args.input)
        if __debug__:
            self.decode_write_fn(img, "/tmp/original.png") # Save a copy for comparing later
            self.output_bytes = 0
        self.img_shape = img.shape
        return img

    def decode_read(self):
        compressed_img = self.decode_read_fn(self.args.input)
        return compressed_img

    def encode_read_fn(self, fn):
        '''Read the image <fn>.'''
        #img = skimage_io.imread(fn) # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
        #img = Image.open(fn) # https://pillow.readthedocs.io/en/stable/handbook/tutorial.html#using-the-image-class
        try:
            input_size = os.path.getsize(fn)
            self.input_bytes += input_size 
            img = cv.imread(fn, cv.IMREAD_UNCHANGED)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        except:
            req = urllib.request.Request(fn, method='HEAD')
            f = urllib.request.urlopen(req)
            input_size = int(f.headers['Content-Length'])
            self.input_bytes += input_size
            img = skimage_io.imread(fn) # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
        logging.info(f"Read {input_size} bytes from {fn} with shape {img.shape} and type={img.dtype}")
        return img

    def decode_read_fn(self, fn_without_extention):
        fn = fn_without_extention + self.file_extension
        input_size = os.path.getsize(fn)
        self.input_bytes += input_size
        logging.info(f"Read {os.path.getsize(fn)} bytes from {fn}")
        data = open(fn, "rb").read()
        return data

    def encode_write(self, compressed_img):
        '''Save to disk the image specified in the class attribute <
        args.output>.'''
        self.encode_write_fn(compressed_img, self.args.output)

    def decode_write(self, img):
        return self.decode_write_fn(img, self.args.output)

    def encode_write_fn(self, data, fn_without_extention):
        data.seek(0)
        fn = fn_without_extention + self.file_extension
        with open(fn, "wb") as output_file:
            output_file.write(data.read())
        self.output_bytes += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn}")

    def decode_write_fn(self, img, fn):
        try:
            skimage_io.imsave(fn, img)
        except Exception as e:
            logging.error(f"Exception \"{e}\" saving image {fn} with shape {img.shape} and type {img.dtype}")
        self.output_bytes += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn} with shape {img.shape} and type {img.dtype}")

