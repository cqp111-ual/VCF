'''Image quantization using a LloydMax quantizer.'''

# Some work could be done with the encoded histograms!

import os
import numpy as np
import gzip
import logging
import main
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser

from scalar_quantization.LloydMax_quantization import LloydMax_Quantizer as Quantizer # pip install "scalar_quantization @ git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"
from scalar_quantization.LloydMax_quantization import name as quantizer_name # pip install "scalar_quantization @ git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"

#from quantization import EC

import entropy_image_coding as EIC
import importlib

default_QSS = 32
default_EIC = "PNG"

parser.parser_encode.add_argument("-e", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)
parser.parser_decode.add_argument("-e", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)
parser.parser_encode.add_argument("-q", "--QSS", type=parser.int_or_str, help=f"Quantization step size (default: {default_QSS})", default=default_QSS)
parser.parser_decode.add_argument("-q", "--QSS", type=parser.int_or_str, help=f"Quantization step size (default: {default_QSS})", default=default_QSS)

args = parser.parser.parse_known_args()[0]
EC = importlib.import_module(args.entropy_image_codec)

class CoDec(EC.CoDec):
    
    def __init__(self, args): # ??
        super().__init__(args)

    '''
    def _compress(self, img):
        k = self.quantize(img)
        #k = k.astype(np.uint8)
        compressed_k = super().compress(k)
        return compressed_k
    '''

    def encode(self):
        img = self.encode_read()
        k = self.quantize(img)
        compressed_k = self.compress(k)
        self.encode_write(compressed_k)
        #rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        #return rate

    def decode(self):
        #k = io.imread(self.args.input)
        compressed_k = self.decode_read()
        k = self.decompress(compressed_k)
        y = self.dequantize(k)
        self.decode_write(y)
        #rate = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        #return rate

    '''
    def _decompress(self, compressed_k):
        k = super().decompress(compressed_k)
        #k = k.astype(np.uint8)
        y = self.dequantize(k)
        return y
    '''

    def quantize(self, img):
        '''Quantize the image.'''
        logging.info(f"QSS = {self.args.QSS}")
        with open(f"{self.args.output}_QSS.txt", 'w') as f:
            f.write(f"{self.args.QSS}")
        self.output_bytes = 1 # We suppose that the representation of the QSS requires 1 byte
        logging.info(f"Written {self.args.output}_QSS.txt")
        if len(img.shape) < 3:
            extended_img = np.expand_dims(img, axis=2)
        else:
            extended_img = img
        k = np.empty_like(extended_img)
        for c in range(extended_img.shape[2]):
            histogram_img, bin_edges_img = np.histogram(extended_img[..., c], bins=256, range=(0, 256))
            logging.info(f"histogram = {histogram_img}")
            histogram_img += 1 # Bins cannot be zeroed
            self.Q = Quantizer(Q_step=self.args.QSS, counts=histogram_img)
            centroids = self.Q.get_representation_levels()
            with gzip.GzipFile(f"{self.args.output}_centroids_{c}.gz", "w") as f:
                np.save(file=f, arr=centroids)
            len_codebook = os.path.getsize(f"{self.args.output}_centroids_{c}.gz")
            logging.info(f"Written {len_codebook} bytes in {self.args.output}_centroids_{c}.gz")
            self.output_bytes += len_codebook
            k[..., c] = self.Q.encode(extended_img[..., c])
        return k

    def dequantize(self, k):
        with open(f"{self.args.input}_QSS.txt", 'r') as f:
            QSS = int(f.read())
        logging.info(f"Read QSS={QSS} from {self.args.output}_QSS.txt")
        if len(k.shape) < 3:
            extended_k = np.expand_dims(k, axis=2)
        else:
            extended_k = k
        y = np.empty_like(extended_k)
        for c in range(y.shape[2]):
            with gzip.GzipFile(f"{self.args.input}_centroids_{c}.gz", "r") as f:
                centroids = np.load(file=f)
            logging.info(f"Read {self.args.input}_centroids_{c}.gz")
            self.Q = Quantizer(Q_step=QSS, counts=np.ones(shape=256))
            self.Q.set_representation_levels(centroids)
            y[..., c] = self.Q.decode(extended_k[..., c])
        return y

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
    logging.info(f"quantizer = {quantizer_name}")
