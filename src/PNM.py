'''Fake entropy coding using Portable aNy Map (PNM). '''

import io
import netpbmfile
import main
import logging
import numpy as np
import cv2 as cv
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

class CoDec(EIC.CoDec):

    def __init__(self, args):
        super().__init__(args)
        self.file_extension = ".pnm"

    def compress(self, img):
        logging.debug(f"img.dtype={img.dtype}")
        assert (img.dtype == np.uint8) or (img.dtype == np.uint16), f"current type = {img.dtype}"
        compressed_img = io.BytesIO()
        netpbmfile.imwrite(compressed_img, img)  # It is not allowed to use netpbmfile.imwrite(file=compressed_img, data=img)
        return compressed_img

    def decompress(self, compressed_img):
        compressed_img = io.BytesIO(compressed_img)
        img = netpbmfile.imread(compressed_img)
        logging.debug(f"img.dtype={img.dtype}")
        return img

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
