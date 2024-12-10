'''Entropy Encoding of images using zlib.'''

import io
import numpy as np
import main
import logging
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser
import entropy_image_coding as EIC

# Default IO images
ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/images/lena.png"
ENCODE_OUTPUT = "/tmp/encoded"
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "/tmp/decoded.png"

# Encoder parser
parser.parser_encode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser.parser_encode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {ENCODE_OUTPUT})", default=f"{ENCODE_OUTPUT}")

# Decoder parser
parser.parser_decode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {DECODE_INPUT})", default=f"{DECODE_INPUT}")
parser.parser_decode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {DECODE_OUTPUT})", default=f"{DECODE_OUTPUT}")    

class CoDec (EIC.CoDec):

    def __init__(self, args):
        super().__init__(args)
        self.file_extension = ".npz"

    def compress(self, img):
        compressed_img = io.BytesIO()
        np.savez_compressed(file=compressed_img, a=img)
        return compressed_img

    def decompress(self, compressed_img):
        compressed_img = io.BytesIO(compressed_img)
        img = np.load(compressed_img)['a']
        #print(type(img), img.shape, img.dtype)
        return img

    def ____encode_write_fn(self, compressed_img, fn):
        '''Write to disk the image <compressed_img> with filename <fn>.'''
        compressed_img.seek(0)
        with open(fn, "wb") as output_file:
            output_file.write(compressed_img.read())
        self.output_bytes += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn}")

    def ____encode(self):
        '''Read an image, compress it with zlib, and save it in the disk.
        '''
        img = self.encode_read()
        compressed_img = self.compress(img)
        #compressed_img = io.BytesIO()
        #np.savez_compressed(file=compressed_img, a=img)
        #compressed_img.seek(0)
        #np.save(file=img_for_disk, arr=img)
        #compressed_img = zlib.compress(img_for_disk, COMPRESSION_LEVEL)
        #print(len(compressed_img.read()))
        #with open("/tmp/1.Zlib", "wb") as output_file:
        #    output_file.write(compressed_img)
        #x = zlib.decompress(compressed_img)
        self.encode_write(compressed_img)
        logging.debug(f"output_bytes={self.output_bytes}, img.shape={img.shape}")
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def ____decode(self):
        '''Read a compressed image, decompress it, and save it.'''
        compressed_img = self.decode_read()
        compressed_img_diskimage = io.BytesIO(compressed_img)
        img = np.load(compressed_img_diskimage)['a']
        #decompressed_data = zlib.decompress(compressed_img)
        #img = io.BytesIO(decompressed_data))
        self.decode_write(img)
        logging.debug(f"output_bytes={self.output_bytes}, img.shape={img.shape}")
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
