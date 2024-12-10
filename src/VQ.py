'''Image (spatial 2D) quantization using a vector quantizer.'''

import os
import numpy as np
import logging
import main
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser
from sklearn import cluster  # pip install scikit-learn

import entropy_image_coding as EIC
import importlib

from information_theory import information  # pip install "information_theory @ git+https://github.com/vicente-gonzalez-ruiz/information_theory"
  
default_block_size = 4
default_EIC = "PNG"
default_N_clusters = 256

parser.parser_encode.add_argument("-e", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)
parser.parser_decode.add_argument("-e", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)
parser.parser_encode.add_argument("-b", "--block_size_VQ", type=parser.int_or_str, help=f"Block size (default: {default_block_size})", default=default_block_size)
parser.parser_decode.add_argument("-b", "--block_size_VQ", type=parser.int_or_str, help=f"Block size (default: {default_block_size})", default=default_block_size)
parser.parser_encode.add_argument("-n", "--N_clusters", type=parser.int_or_str, help=f"Number of clusters (default: {default_N_clusters})", default=default_N_clusters)
parser.parser_decode.add_argument("-n", "--N_clusters", type=parser.int_or_str, help=f"Number of clusters (default: {default_N_clusters})", default=default_N_clusters)

args = parser.parser.parse_known_args()[0]
EC = importlib.import_module(args.entropy_image_codec)

class CoDec(EC.CoDec):

    def __init__(self, args, min_index_val=0, max_index_val=255):
        super().__init__(args)
        logging.debug(f"args = {self.args}")
        self.BS = args.block_size_VQ
        self.N_clusters = args.N_clusters
        self.input = args.input
        self.output = args.output

    def encode(self):
        img = self.encode_read()
        #k, centroids = self.quantize(img)
        k = self.quantize(img)
        compressed_k = self.compress(k)
        self.encode_write(compressed_k)
        #self.encode_write_fn(compressed_k, self.output + "_labels")
        #compressed_centroids = self.compress(centroids)
        #self.encode_write_fn(compressed_centroids, self.output + "_centroids")
    
    def decode(self):
        compressed_k = self.decode_read()
        #compressed_centroids = self.decode_read_fn(self.input + "_centroids")
        #compressed_k = self.decode_read_fn(self.input + "_labels")
        #centroids = self.decompress(compressed_centroids)
        k = self.decompress(compressed_k)
        #y = self.dequantize(k, centroids)
        y = self.dequantize(k)
        self.decode_write(y)

    def quantize(self, img):
        blocks = []
        BL = self.BS * self.BS * img.shape[2]
        for y in range(0, img.shape[0], self.BS):
            for x in range(0, img.shape[1], self.BS):
                blocks.append(np.reshape(img[y:y + self.BS,
                                             x:x + self.BS], BL))
        blocks = np.asarray(blocks).astype(int)
        if(len(blocks) < self.N_clusters):
            logging.warning('\033[91m' + "Warning: Must reduce number of clusters. Image not big enough | too much pixel reducction" + '\033[0m')
            self.N_clusters = len(blocks)
        #initial_centroids = np.ones(shape=(self.N_clusters, self.BS*self.BS*img.shape[2]))*255
        #for i in range(self.N_clusters): # Ojo, que quizás no se use
        #    initial_centroids[i] = np.round(initial_centroids[i]/self.N_clusters)
        #k_means = cluster.KMeans(init=initial_centroids, n_clusters=self.N_clusters, n_init=1)
        k_means = cluster.KMeans(init="k-means++", n_clusters=self.N_clusters, n_init=1)
        #k_means = cluster.KMeans(init=initial_centroids, n_init=1, n_clusters=self.N_clusters, random_state=0, algorithm="elkan")
        
        k_means.fit(blocks)
        centroids = k_means.cluster_centers_#.astype(np.uint8)
        centroids_energy = np.empty(centroids.shape[0])
        counter = 0
        for i in centroids:
            centroids_energy[counter] = information.energy(i)
            counter += 1
        argsort_centroids = np.argsort(centroids_energy)
        lut = np.empty_like(argsort_centroids, dtype=np.int16)
        lut[argsort_centroids] = np.arange(self.N_clusters)
        labels = k_means.labels_
        labels = lut[labels]
        _centroids = np.empty_like(centroids)
        for i in range(self.N_clusters):
            _centroids[lut[i]] = centroids[i]
        centroids = _centroids
        labels_shape = (img.shape[0]//self.BS,
                        img.shape[1]//self.BS)
        #labels = labels.astype(np.uint8).reshape(labels_shape)
        labels = labels.reshape(labels_shape)
        #print("--------", np.min(labels), np.max(labels))
        labels = labels.astype(np.uint16)
        centroids = centroids.reshape(centroids.shape[0], self.BS*self.BS, img.shape[2])#.astype(np.uint8)
        #compressed_centroids = self.compress(centroids)
        #self.encode_write_fn(compressed_centroids, self.output + "_centroids")
        fn = self.output + "_centroids.npz"
        np.savez_compressed(file=fn, a=centroids)
        self.output_bytes += os.path.getsize(fn)
        return labels
        #return labels, centroids

    #def dequantize(self, labels, centroids):
    def dequantize(self, labels):
        #compressed_centroids = self.decode_read_fn(self.input + "_centroids")
        #centroids = self.decompress(compressed_centroids)
        fn = self.input + "_centroids.npz"
        self.input_bytes += os.path.getsize(fn)
        centroids = np.load(file=fn)['a']
        img_shape = (labels.shape[0]*self.BS, labels.shape[1]*self.BS, centroids.shape[2])
        _y = np.empty(shape=(labels.shape[0]*self.BS,
                             labels.shape[1]*self.BS,
                             centroids.shape[2]), dtype=np.uint8)
        for y in range(0, img_shape[0], self.BS):
            for x in range(0, img_shape[1], self.BS):
                #print("---", labels)
                _y[y:y + self.BS,
                   x:x + self.BS, :] = \
                centroids[labels[y//self.BS,
                                 x//self.BS]].reshape(self.BS,
                                                      self.BS,
                                                      centroids.shape[2])
        return _y

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
