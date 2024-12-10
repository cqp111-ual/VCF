'''Entropy Encoding of images non-adaptive Huffman Coding'''

import io
import numpy as np
import main
import logging
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser
import entropy_image_coding as EIC
import heapq
from collections import defaultdict, Counter
import gzip
import pickle
from bitarray import bitarray
import os
import math

# Default IO images
ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/images/lena.png"
ENCODE_OUTPUT = "/tmp/encoded" # The file extension is decided in run-time
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "/tmp/decoded.png"

# Encoder parser
parser.parser_encode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser.parser_encode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {ENCODE_OUTPUT})", default=f"{ENCODE_OUTPUT}")

# Decoder parser
parser.parser_decode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {DECODE_INPUT})", default=f"{DECODE_INPUT}")
parser.parser_decode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {DECODE_OUTPUT})", default=f"{DECODE_OUTPUT}")    

#parser.parser.parse_known_args()

class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    frequency = Counter(data)
    heap = [HuffmanNode(value, freq) for value, freq in frequency.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    
    return heap[0]

def generate_huffman_codes(node, current_code="", codes={}):
    if node is None:
        return
    if node.value is not None:
        codes[node.value] = current_code
    generate_huffman_codes(node.left, current_code + "0", codes)
    generate_huffman_codes(node.right, current_code + "1", codes)
    return codes

def encode_data(data, codes):
    # Create a single concatenated string of all encoded bits
    encoded_string = ''.join(codes[value] for value in data)
    #print("-------------_", len(data))
    # Convert this string of bits to a bitarray
    encoded_data = bitarray(encoded_string)

    return encoded_data

def decode_data(encoded_data, root):
    data = []
    node = root
    for bit in encoded_data:
        if bit == 0:
            node = node.left
        else:
            node = node.right
        # If it's a leaf node, record the value and reset to root
        if node.left is None and node.right is None:
            data.append(node.value)
            node = root
    #print("-------------_", len(data))
    return data

class CoDec (EIC.CoDec):

    def __init__(self, args):
        super().__init__(args)
        self.file_extension = ".huf"

    def compress(self, img):
        tree_fn = f"{self.args.output}_huffman_tree.pkl.gz"
        compressed_img = io.BytesIO()

        # Flatten the array and convert to a list
        flattened_img = img.flatten().tolist()

        # Build Huffman Tree and generate the Huffman codes
        root = build_huffman_tree(flattened_img)
        codes = generate_huffman_codes(root)

        # Encode the flattened array
        encoded_img = encode_data(flattened_img, codes)

        # Write encoded image and original shape to compressed_img
        compressed_img.write(encoded_img.tobytes())  # Save encoded data as bytes

        # Compress and save shape and the Huffman Tree
        with gzip.open(tree_fn, 'wb') as f:
            np.save(f, img.shape)
            pickle.dump(root, f)  # `gzip.open` compresses the pickle data

        tree_length = os.path.getsize(tree_fn)
        logging.info(f"Length of the file \"{tree_fn}\" (Huffman tree + image shape) = {tree_length} bytes")
        self.output_bytes += tree_length

        return compressed_img
    
    def decompress(self, compressed_img):
        tree_fn = f"{self.args.input}_huffman_tree.pkl.gz"
        compressed_img = io.BytesIO(compressed_img)
        
        # Load the shape and the Huffman Tree from the compressed file
        with gzip.open(tree_fn, 'rb') as f:
            shape = np.load(f)
            root = pickle.load(f)
    
        # Read encoded image data as binary
        encoded_data = bitarray()
        encoded_data.frombytes(compressed_img.read())
    
        # Decode the image
        decoded_data = decode_data(encoded_data, root)
        if math.prod(shape) < len(decoded_data):
            decoded_data = decoded_data[:math.prod(shape) - len(decoded_data)] # Sometimes, when the alphabet size is small, some extra symbols are decoded :-/

        # Reshape decoded data to original shape
        img = np.array(decoded_data).reshape(shape).astype(np.uint8)
        return img

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)





