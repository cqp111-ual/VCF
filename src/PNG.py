'''Entropy Encoding of images using PNG (Portable Network Graphics). '''

import io
from skimage import io as skimage_io # pip install scikit-image
import main
import logging
import numpy as np
import cv2 as cv # pip install opencv-python
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser
import entropy_image_coding as EIC

# Default IO images
ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/images/lena.png"
ENCODE_OUTPUT = "/tmp/encoded" # The file extension is decided in run-time
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "/tmp/decoded.png"

#_parser, parser_encode, parser_decode = parser.create_parser(description=__doc__)

# Encoder parser
parser.parser_encode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser.parser_encode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {ENCODE_OUTPUT})", default=f"{ENCODE_OUTPUT}")

# Decoder parser
parser.parser_decode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {DECODE_INPUT})", default=f"{DECODE_INPUT}")
parser.parser_decode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {DECODE_OUTPUT})", default=f"{DECODE_OUTPUT}")    

#parser.parser.parse_known_args()

COMPRESSION_LEVEL = 9

class CoDec(EIC.CoDec):

    def __init__(self, args):
        super().__init__(args)
        self.file_extension = ".png"

    # pip install imageio-freeimage (not necessary now)
    def compress(self, img):
        #skimage_io.use_plugin('freeimage')
        #compressed_img = img
        logging.debug(f"img.dtype={img.dtype}")
        assert (img.dtype == np.uint8) or (img.dtype == np.uint16), f"current type = {img.dtype}"
        #assert (img.dtype == np.uint8), f"current type = {img.dtype}"
        compressed_img = io.BytesIO()
        skimage_io.imsave(fname=compressed_img, arr=img, plugin="pil", check_contrast=False)
        #skimage_io.imsave(fname=compressed_img, arr=img, plugin="freeimage")
        #img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        #cv.imwrite(compressed_img, img, [cv.IMWRITE_PNG_COMPRESSION, COMPRESSION_LEVEL])
        return compressed_img

    def decompress(self, compressed_img):
        compressed_img = io.BytesIO(compressed_img)
        #img = cv.imread(compressed_img, cv.IMREAD_UNCHANGED)
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = skimage_io.imread(fname=compressed_img)
        logging.debug(f"img.dtype={img.dtype}")
        return img

    def _encode_write_fn(self, img, fn):
        '''Write to disk the image <img> with filename <fn>.'''
        skimage_io.imsave(fn, img)
        self.output_bytes += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn} with shape {img.shape} and type {img.dtype}")

    def _write_fn(self, img, fn):
        '''Write to disk the image with filename <fn>.'''
        # Notice that the encoding algorithm depends on the output
        # file extension (PNG).
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite(fn, img, [cv.IMWRITE_PNG_COMPRESSION, COMPRESSION_LEVEL])
        #if __debug__:
        #    len_output = os.path.getsize(fn)
        #    logging.info(f"Before optipng: {len_output} bytes")
        #subprocess.run(f"optipng {fn}", shell=True, capture_output=True)
        self.output_bytes += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn} with shape {img.shape} and type {img.dtype}")

    def _write_fn(self, img, fn):
        '''Write to disk the image with filename <fn>.'''
        # Notice that the encoding algorithm depends on the output
        # file extension (PNG).
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite(fn, img, [cv.IMWRITE_PNG_COMPRESSION, COMPRESSION_LEVEL])

        #skimage_io.imsave(fn, img, check_contrast=False)
        #image = Image.fromarray(img.astype('uint8'), 'RGB')
        #image.save(fn)
        #subprocess.run(f"optipng -nc {fn}", shell=True, capture_output=True)
        subprocess.run(f"pngcrush {fn} /tmp/pngcrush.png", shell=True, capture_output=True)
        subprocess.run(f"mv -f /tmp/pngcrush.png {fn}", shell=True, capture_output=True)
        # Notice that pngcrush is not installed, these two previous steps do not make any effect!
        self.output_bytes += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn} with shape {img.shape} and type {img.dtype}")

    def _decode(self):
        '''Read an image and save it in the disk. Notice that we are
        using the PNG image format for both, decode and encode an
        image. For this reason, both methods do exactly the same.
        This method is overriden in child classes.

        '''
        return self.encode()

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
