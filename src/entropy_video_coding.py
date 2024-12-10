'''Shared code among the video entropy codecs. "Uncompressed" (i.e.,
the input of encode and the output of decode) IO uses lossless H.264
encapsulated in AVI. Videos are not loaded to memory, but only the
required images.

'''

import os
import io
from skimage import io as skimage_io # pip install scikit-image
from PIL import Image # pip install 
import numpy as np
import logging
import subprocess
import cv2
import main
import urllib
from urllib.parse import urlparse
#import requests
import av

from information_theory import distortion # pip install "information_theory @ git+https://github.com/vicente-gonzalez-ruiz/information_theory"

class Video:
    '''A video is a sequence of files stored in "prefix".'''

    def __init__(self, N_frames, height, width, fn):
        self.N_frames = N_frames
        self.height = height
        self.width = width
        self.fn = fn

    def get_shape(self):
        return self.N_frames, self.height, self.width

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
        self.framerate = 30

    def __del__(self):
        logging.info(f"Total {self.input_bytes} bytes read")
        logging.info(f"Total {self.output_bytes} bytes written")
        logging.info(f"N_frames={self.N_frames}")
        logging.info(f"width={self.width}")
        logging.info(f"height={self.height}")
        logging.info(f"N_channels={self.N_channels}")
        if __debug__:
            if self.encoding:
                BPP = (self.output_bytes*8)/(self.N_frames*self.width*self.height)
                logging.info(f"Encoding (output) rate = {BPP} bits/pixel")
                with open(f"{self.args.output}.txt", 'w') as f:
                    f.write(f"{self.args.input}\n")
                    f.write(f"{self.N_frames}\n")
                    f.write(f"{BPP}\n")
            else:
                with open(f"{self.args.input}.txt", 'r') as f:
                    original_file = f.readline().strip()
                    logging.info(f"original_file={original_file}")
                    N_frames = float(f.readline().strip())
                    logging.info(f"N_frames={N_frames}")
                    BPP = float(f.readline().strip())
                    logging.info(f"BPP={BPP}")
                container_x = av.open(original_file)
                container_y = av.open(self.args.output)
                img_counter = 0
                total_RMSE = 0
                logging.info("Computing RD performance ...")
                for frame_x, frame_y in zip(container_x.decode(video=0), container_y.decode(video=0)):
                    img_x = np.array(frame_x.to_image())
                    img_y = np.array(frame_y.to_image())
                    img_RMSE = distortion.RMSE(img_x, img_y)
                    #print(img_RMSE)
                    total_RMSE += img_RMSE
                    print(f"{img_counter}/{self.N_frames}", end='\r', flush=True)
                    img_counter += 1
                RMSE = total_RMSE / self.N_frames
                logging.info(f"\nRMSE = {RMSE}")

                J = BPP + RMSE
                logging.info(f"J = R + D = {J}")

    def encode(self):
        #vid = self.encode_read()
        #compressed_vid = self.compress(vid)
        self.compress()
        #self.shape = compressed_vid.get_shape()
        #self.encode_write(compressed_vid)

    def _encode_read(self):
        '''"Read" the video specified in the class attribute args.input.'''
        vid = self.encode_read_fn(self.args.input)
        if __debug__:
            self.decode_write_fn(vid, "/tmp/original.avi") # Save a copy for comparing later
            self.output_bytes = 0
        return vid

    def __is_http_url(self, url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme.lower() in ['http', 'https']
        except ValueError:
            return False
        
    def _encode_read_fn(self, fn):
        '''"Read" the video <fn>, which can be a URL. The video is
        saved in "/tmp/<fn>".'''

        from urllib.parse import urlparse
        import av
    
        if self.__is_http_url(fn):
            response = requests.get(fn, stream=True)
            if response.status_code == 200: # If the download was successful
                input_size = 0
                #file_path = os.path.join("/tmp", fn)
                file_path = "/tmp/original.avi"
                with open(file_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            input_size += 8192
                            print('.', end='', flush=True)
                print("\nVideo downloaded")
                #fn = io.BytesIO(response.content) # Open the downloaded video as a byte stream
                #input_size = len(fn)
                #req = urllib.request.Request(fn, method='HEAD')
                #f = urllib.request.urlopen(req)
                #input_size = int(f.headers['Content-Length'])
        else:
            input_size = os.path.getsize(fn)
        self.input_bytes += input_size

        cap = cv2.VideoCapture(fn)
        N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(fn, N_frames)
        #digits = len(str(N_frames))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if __debug__:
            self.N_frames = N_frames
            self.shape = (N_frames, height, width)

        vid = Video(N_frames, height, width, fn)
        logging.info(f"Read {input_size} bytes from {fn} with shape {vid.get_shape()[1:]}")

        return vid

    def _encode_write(self, compressed_vid):
        '''Save to disk the video specified in the class attribute args.output.'''
        self.encode_write_fn(compressed_vid, self.args.output)

    def _encode_write_fn(self, data, fn_without_extention):
        #data.seek(0)
        fn = fn_without_extention + self.file_extension
        with open(fn, "wb") as output_file:
            output_file.write(data.read())
        self.output_bytes += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn}")

    def decode(self):
        #compressed_vid = self.decode_read()
        #vid = self.decompress(compressed_vid)
        self.decompress()
        #self.decode_write(vid)

    def _decode_read(self):
        compressed_vid = self.decode_read_fn(self.args.input)
        return compressed_vid

    def _decode_write(self, vid):
        return self.decode_write_fn(vid, self.args.output)

    def _decode_read_fn(self, fn_without_extention):
        fn = fn_without_extention + self.file_extension
        input_size = os.path.getsize(fn)
        self.input_bytes += input_size
        logging.info(f"Read {os.path.getsize(fn)} bytes from {fn}")
        data = open(fn, "rb").read()
        return data

    def _decode_write_fn(self, vid, fn):
        pass
        '''
        frames = [e for e in os.listdir(vid.prefix)]
        for i in frames:
            skimage_io.imsave(fn, img)
        self.output_bytes += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn} with shape {img.shape} and type {img.dtype}")
        '''

###################################################

    def _encode_read_fn(self, fn):
        '''Read the video <fn>.'''

    
        if __is_http_url(fn):
            response = requests.get(fn) # Download the video file (in memory)
            if response.status_code == 200: # If the download was successful
                fn = BytesIO(response.content) # Open the downloaded video as a byte stream
                input_size = len(fn)
        else:
            input_size = os.path.getsize(fn)
        self.input_bytes += input_size 
        cap = cv2.VideoCapture(fn)
        N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        digits = len(str(N_frames))
        img_counter = 0
        while True:
            ret, img = cap.read()

            if not ret:
                break # Break the loop if the video has ended

            # Write the frame in /tmp/VCF_input
            img_fn = os.path.join("/tmp", f"img_{img_counter:0{digits}d}.png")
            img_counter += 1
            cv2.imwrite(img_fn, img)
        return Video(N_frames, img.shape[0], img.shape[1], "/tmp/img_")

    def _encode_read_fn(self, fn):
        '''Read the video <fn>.'''

        from urllib.parse import urlparse
        import imageio_ffmpeg as ffmpeg
    
        if __is_http_url(fn):
            response = requests.get(fn) # Download the video file (in memory)
            if response.status_code == 200: # If the download was successful
                fn = BytesIO(response.content) # Open the downloaded video as a byte stream
                input_size = len(fn)
        else:
            input_size = os.path.getsize(fn)
        self.input_bytes += input_size

        cap = cv2.VideoCapture(fn)
        N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        digits = len(str(N_frames))

        with ffmpeg.get_reader(fn) as reader:
            for i, img in enumerate(reader):
                frame_array = np.array(img)        

                # Write the frame in /tmp/img_
                img_fn = os.path.join("/tmp", f"img_{img_counter:0{digits}d}.png")
                img_counter += 1
                cv2.imwrite(img_fn, img)
            N_frames = len(reader)
            logging.info(f"")
        return Video(N_frames, img.shape[0], img.shape[1], "/tmp/frame_")

