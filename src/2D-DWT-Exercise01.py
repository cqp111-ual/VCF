'''Exploiting spatial redundancy with the 2D dyadic Discrete Wavelet Transform.'''

import io
from skimage import io as skimage_io # pip install scikit-image
import numpy as np
import pywt # pip install pywavelets
import os
import logging
import main
with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)
import parser
import importlib

import re # Added

#from DWT import color_dyadic_DWT as DWT
from DWT2D.color_dyadic_DWT import analyze as space_analyze # pip install "DWT2D @ git+https://github.com/vicente-gonzalez-ruiz/DWT2D"
from DWT2D.color_dyadic_DWT import synthesize as space_synthesize

from color_transforms.YCoCg import from_RGB # pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import to_RGB

default_levels = 5
default_DWT = "db5"
default_CT = "YCoCg"

parser.parser_encode.add_argument("-l", "--levels", type=parser.int_or_str, help=f"Number of decomposition levels (default: {default_levels})", default=default_levels)
parser.parser_encode.add_argument("-w", "--wavelet", type=parser.int_or_str, help=f"Wavelet name (default: \"{default_DWT}\")", default=default_DWT)
parser.parser_encode.add_argument("-t", "--color_transform", type=parser.int_or_str, help=f"Color transform (default: \"{default_CT}\")", default=default_CT)
parser.parser_decode.add_argument("-l", "--levels", type=parser.int_or_str, help=f"Number of decomposition levels (default: {default_levels})", default=default_levels)
parser.parser_decode.add_argument("-w", "--wavelet", type=parser.int_or_str, help=f"Wavelet name (default: \"{default_DWT}\")", default=default_DWT)
parser.parser_decode.add_argument("-t", "--color_transform", type=parser.int_or_str, help=f"Color transform (default: \"{default_CT}\")", default=default_CT)

#import PNG as EC
#import YCoCg as CT # Color Transform

args = parser.parser.parse_known_args()[0]
CT = importlib.import_module(args.color_transform)

class CoDec(CT.CoDec):

    def __init__(self, args):
        super().__init__(args)
        # Get debug param
        self.debug = args.debug
        # Get operation param
        self.operation = args.subparser_name
        self.levels = args.levels
        if self.operation == "decode":
            # List files in tmp and get highest encoded_LL_X.png
            encoded_level = self.get_encoded_level()
            # Check if levels parameter is valid, should be between 1 and encoded level
            if 1<= self.levels and self.levels <= encoded_level:
                self.decode_level = self.levels
                self.levels = encoded_level
            else:
                logging.warning(f"Levels parameter is out of range. Should between 1 and {encoded_level}")
                self.levels = encoded_level
                self.decode_level = encoded_level
                logging.warning(f"Readjusting parameter: levels = {self.levels}")

            logging.info(f"decode_level={self.decode_level}")
        logging.info(f"levels = {self.levels}")

        self.wavelet = pywt.Wavelet(args.wavelet)
        logging.info(f"wavelet={args.wavelet} ({self.wavelet})")

    # Get encoding level, necessary to decode image
    def get_encoded_level(self):
        tmp_dir = "/tmp"
        pattern = r"encoded_LL_(\d+)\.png"
        encoded_level = None

        # Reads /tmp directory searching for the highest encoded_LL_{number}
        try:
            for filename in os.listdir(tmp_dir):
                match = re.match(pattern, filename)
                if match:
                    level = int(match.group(1))
                    if encoded_level is None or level > encoded_level:
                        encoded_level = level
        except Exception as e:
            logging.error(f"Error reading files in {tmp_dir}: {e}")
            return None

        logging.debug(f"Levels when encoded: levels={encoded_level}")
        return encoded_level

    def encode(self):
        img = self.encode_read().astype(np.int16)
        img_128 = img #- 128 # To use the deadzone
        CT_img = from_RGB(img_128)
        
        decom_img = space_analyze(CT_img, self.wavelet, self.levels)
        logging.debug(f"len(decom_img)={len(decom_img)}")
        decom_k = self.quantize_decom(decom_img)
        self.write_decom(decom_k)

        #k = self.quantize(CT_img)
        #logging.debug(f"k.shape={k.shape}, k.type={k.dtype}")
        #k[..., 1] += 128
        #k[..., 2] += 128
        #compressed_k = self.compress(k.astype(np.uint8))
        #self.encode_write(compressed_k)

        self.BPP = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        #return rate

    def decode(self):
        # Reads all subbands
        decom_k = self.read_decom()

        # To obtain a custom decoded level simply ignore 
        # subbands higher than self.decoded_level

        if self.debug:
            print(f"[Before]")
            print(f"len(decom_k) = {len(decom_k)}")
            print(f"decom_k[0] = [LL{self.levels}]")
            for i, elem in enumerate(decom_k[1:], start=1):
                # print(f"Elemento {i}: Tipo={type(elem)}, Longitud={len(elem) if isinstance(elem, (list, np.ndarray)) else 'N/A'}")
                print(f"decom_k[{i}] = [LH{self.levels-i+1}, HL{self.levels-i+1}, HH{self.levels-i+1}]")

        # decom_y = decom_k (Before)
        decom_y=decom_k[:self.decode_level+1]
        decom_k=decom_y

        if self.debug:
            print(f"[After]")
            print(f"len(decom_k) = {len(decom_k)}")
            print(f"decom_k[0] = [LL{self.levels}]")
            for i, elem in enumerate(decom_k[1:], start=1):
                # print(f"Elemento {i}: Tipo={type(elem)}, Longitud={len(elem) if isinstance(elem, (list, np.ndarray)) else 'N/A'}")
                print(f"decom_k[{i}] = [LH{self.levels-i+1}, HL{self.levels-i+1}, HH{self.levels-i+1}]")

        decom_y = self.dequantize_decom(decom_k)

        # Adjust last paramater, to avoid out of range
        # CT_y = space_synthesize(decom_y, self.wavelet, self.levels)
        CT_y = space_synthesize(decom_y, self.wavelet, len(decom_y)-1)

        y_128 = to_RGB(CT_y)
        y = y_128# + 128
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.decode_write(y)

        self.BPP = (self.input_bytes*8)/(y.shape[0]*y.shape[1])
        #return rate

    def quantize_decom(self, decom):
        LL_k = super().quantize(decom[0])
        LL_k[..., 1] += 128
        LL_k[..., 2] += 128
        decom_k = [LL_k]
        for spatial_resolution in decom[1:]:
            spatial_resolution_k = []
            for subband in spatial_resolution:
                subband_k = super().quantize(subband)
                subband_k += 128
                spatial_resolution_k.append(subband_k)
            decom_k.append(tuple(spatial_resolution_k))
        return decom_k

    def dequantize_decom(self, decom_k):
        LL_k = decom_k[0]
        LL_k[..., 1] -= 128
        LL_k[..., 2] -= 128
        decom_y = [super().dequantize(LL_k)]
        for spatial_resolution_k in decom_k[1:]:
            spatial_resolution_y = []
            for subband_k in spatial_resolution_k:
                subband_k -= 128
                subband_y = super().dequantize(subband_k)
                spatial_resolution_y.append(subband_y)
            decom_y.append(tuple(spatial_resolution_y))
        return decom_y

    def _quantize_decom(self, decom):
        decom_k = [self.quantize(decom[0])] # LL subband
        for spatial_resolution in decom[1:]:
            spatial_resolution_k = []
            for subband in spatial_resolution:
                subband_k = self.quantize(subband)
                spatial_resolution_k.append(subband_k)
            decom_k.append(tuple(spatial_resolution_k))
        return decom_k

    def _dequantize_decom(self, decom_k):
        decom_y = [self.dequantize(decom_k[0])]
        for spatial_resolution_k in decom_k[1:]:
            spatial_resolution_y = []
            for subband_k in spatial_resolution_k:
                subband_y = self.dequantize(subband_k)
                spatial_resolution_y.append(subband_y)
            decom_y.append(tuple(spatial_resolution_y))
        return decom_y

    def _quantize(self, subband):
        '''Quantize the image.'''
        #k = self.Q.encode(subband)
        #k = super().quantize(subband)
        k = subband
        k += 32768
        k = k.astype(np.uint16)
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype}")
        return k

    def _dequantize(self, k):
        '''"Dequantize" an image.'''
        k = k.astype(np.int16)
        k -= 32768
        #self.Q = Quantizer(Q_step=QSS, min_val=min_index_val, max_val=max_index_val)
        #y = self.Q.decode(k)
        #y = super().dequantize(k)
        y = k
        logging.debug(f"y.shape={y.shape} y.dtype={y.dtype}")
        return y

    def write_decom(self, decom):
        LL = decom[0]
        fn_without_extension = self.args.output.split('.')[0]
        fn_subband = f"{fn_without_extension}_LL_{self.levels}"
        #LL = io.BytesIO(LL)
        LL = self.compress(LL.astype(np.uint8))
        self.encode_write_fn(LL, fn_subband)
        resolution_index = self.levels
        #aux_decom = [decom[0][..., 0]] # Used for computing slices
        for spatial_resolution in decom[1:]:
            subband_names = ["LH", "HL", "HH"]
            subband_index = 0
            #aux_resol = [] # Used for computing slices
            for subband_name in subband_names:
                fn_subband = f"{fn_without_extension}_{subband_name}_{resolution_index}"
                #SP = io.BytesIO(spatial_resolution[subband_index])
                SP = self.compress(spatial_resolution[subband_index].astype(np.uint8))
                self.encode_write_fn(SP, fn_subband)
                #aux_resol.append(spatial_resolution[subband_index][..., 0])
                subband_index += 1
            resolution_index -= 1
            #aux_decom.append(tuple(aux_resol))
        #self.slices = pywt.coeffs_to_array(aux_decom)[1]
        #return slices

    def read_decom(self):
        fn_without_extension = self.args.input.split('.')[0]
        fn_subband = f"{fn_without_extension}_LL_{self.levels}"
        LL = self.decode_read_fn(fn_subband)
        LL = self.decompress(LL).astype(np.int16)
        decom = [LL]
        resolution_index = self.levels
        for l in range(self.levels, 0, -1):
            subband_names = ["LH", "HL", "HH"]
            spatial_resolution = []
            for subband_name in subband_names:
                fn_subband = f"{fn_without_extension}_{subband_name}_{resolution_index}"
                subband = self.decode_read_fn(fn_subband)
                subband = self.decompress(subband).astype(np.int16)
                spatial_resolution.append(subband)
            decom.append(tuple(spatial_resolution))
            resolution_index -= 1
        return decom

    '''
    def __save_fn(self, img, fn):
        io.imsave(fn, img, check_contrast=False)
        self.required_bytes = os.path.getsize(fn)
        logging.info(f"Written {self.required_bytes} bytes in {fn}")

    def __read_fn(self, fn):
        img = io.imread(fn)
        logging.info(f"Read {fn} of shape {img.shape}")
        return img
    '''

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)