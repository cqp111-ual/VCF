'''Exploiting color (perceptual) redundancy with the YCoCg transform.'''

import numpy as np
import logging
import main
import importlib
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser

from color_transforms.YCoCg import from_RGB # pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import to_RGB
#from information_theory import distortion # pip install "information_theory @ git+https://github.com/vicente-gonzalez-ruiz/information_theory"

default_quantizer = "deadzone"

#_parser, parser_encode, parser_decode = parser.create_parser(description=__doc__)

parser.parser_encode.add_argument("-c", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)
parser.parser_decode.add_argument("-c", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)

args = parser.parser.parse_known_args()[0]
Q = importlib.import_module(args.quantizer)

class CoDec(Q.CoDec):

    def __init__(self, args):
        super().__init__(args)
        if args.quantizer == "deadzone":
            self.offset = 128
        else:
            self.offset = 0

    def _compress(self, img):
        img = img.astype(np.int16)
        img -= 128
        YCrCb_img = from_RGB(img)
        compressed_k = super().compress(YCrCb_img)
        return compressed_k

    def _decompress(self, compressed_k):
        YCrCb_y = super().decompress(compressed_k)
        y = to_RGB(YCrCb_y)
        y = np.clip(y, 0, 255)
        y = y.astype(np.uint8)
        return y

    def encode(self):
        img = self.encode_read()
        img = img.astype(np.int16)
        # Specific for solving the issue https://github.com/vicente-gonzalez-ruiz/scalar_quantization/issues/1
        #img_128 = img.astype(np.int16) - 128
        #YCoCg_img_128 = from_RGB(img_128)
        #YCoCg_img = YCoCg_img_128 + 128
        img -= self.offset
        YCoCg_img = from_RGB(img)
        #YCoCg_img[..., 1] += 128
        #YCoCg_img[..., 2] += 128
        #logging.debug(f"max(YCoCg_img)={np.max(YCoCg_img)}, min(YCoCg_img)={np.min(YCoCg_img)}")
        #assert (YCoCg_img < 256).all()
        #assert (YCoCg_img >= 0).all()
        k = self.quantize(YCoCg_img)
        logging.debug(f"k.shape={k.shape}, k.type={k.dtype}")
        #k = YCoCg_img
        #k[..., 1] += 128
        #k[..., 2] += 128
        k += self.offset
        if np.max(k) > 255:
            logging.warning(f"k[{np.unravel_index(np.argmax(k),k.shape)}]={np.max(k)}")
        if np.min(k) < 0:
            logging.warning(f"k[{np.unravel_index(np.argmin(k),k.shape)}]={np.min(k)}")
        k = np.clip(k, 0, 255).astype(np.uint8)
        compressed_k = self.compress(k)
        self.encode_write(compressed_k)
        #self.BPP = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        #logging.info(f"BPP = {BPP}")

    def decode(self):
        compressed_k = self.decode_read()
        k = self.decompress(compressed_k)
        k = k.astype(np.int16)
        logging.debug(f"k.shape={k.shape}, k.type={k.dtype}")
        k -= self.offset
        #k[..., 1] -= 128
        #k[..., 2] -= 128
        YCoCg_y = self.dequantize(k)
        #YCoCg_y = k
#        logging.debug(f"max(YCoCg_y)={np.max(YCoCg_y)}, min(YCoCg_y)={np.min(YCoCg_y)}")
#        assert (YCoCg_y < 256).all()
#        assert (YCoCg_y >= 0).all()
#        logging.debug(f"YCoCg_y.shape={YCoCg_y.shape}, YCoCg_y.type={YCoCg_y.dtype}")
        # Specific for solving the issue https://github.com/vicente-gonzalez-ruiz/scalar_quantization/issues/1
        #YCoCg_y_128 = YCoCg_y.astype(np.int16) - 128
        #y_128 = to_RGB(YCoCg_y_128)
        #y = y_128 + 128
        #YCoCg_y[..., 1] -= 128
        #YCoCg_y[..., 2] -= 128        
        y = to_RGB(YCoCg_y)
        y += self.offset
        logging.debug(f"y.shape={y.shape}, y.type={y.dtype}")
        if np.max(y) > 255:
            logging.warning(f"y[{np.unravel_index(np.argmax(y),y.shape)}]={np.max(y)}")
        if np.min(y) < 0:
            logging.warning(f"y[{np.unravel_index(np.argmin(y),y.shape)}]={np.min(y)}")
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.decode_write(y)
        #self.BPP = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        #RMSE = distortion.RMSE(self.encode_read(), y)
        #logging.info(f"RMSE = {RMSE}")

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
