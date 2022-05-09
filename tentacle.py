#!/usr/bin/env python3

"""
OPENMMLAB's MMOCR INFERENCE CLIENT
==================================

The following program is used to perform inference on the subject data
"""

# %%
# Importing Libraries
import numpy as np
import json
import grpc
import communication_pb2
import communication_pb2_grpc

# %%
# Main Client Inference Class
class mmocr_ocr:
    def __init__(self, server_ip, id_len = 10):
        """
        This method is used to initialize MMOCR inference client

        Method Input
        =============
        server_ip : Server IP at which GRPC server is running
                            Format : "IP:Port"
                            Example : '0.0.0.0:4321'
        id_len : Length of client randomized id ( default : 10 )

        Method Output
        ==============
        None
        """
        self.mmocr_server_ip = server_ip
        self.channel = grpc.insecure_channel(self.mmocr_server_ip)
        self.stub = communication_pb2_grpc.mmocr_serviceStub(self.channel)
        self.client_name_chars = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
        self.client_name = ''.join(np.random.choice(self.client_name_chars, size = id_len).tolist())

    def input_processor(self, inp1):
        """
        This method is used to process input & sends request to GRPC server

        Method Input
        =============
        inp1 : Input for inference request as Numpy array
                            [ Batch x Width x Height x Channel ]

        Method Output
        ==============
        Request to GRPC server
        """
        inp1_shape = inp1.shape
        self.__batch_size = inp1_shape[0]
        return communication_pb2.server_input(
            imgs = inp1.tobytes(),
            batch = inp1_shape[0],
            width = inp1_shape[1],
            height = inp1_shape[2],
            channel = inp1_shape[3],
            data_type = inp1.dtype.name,
            client_id = self.client_name)
    
    def output_processor(self, out1):
        """
        This method is used to process output by receiving response from GRPC server

        Method Input
        =============
        out1 : GRPC server response after inference

        Method Output
        ==============
        Response from GRPC server
        """
        return json.loads(out1.results_dictionary.decode())
    
    def __call__(self, x):
        """
        This method is used to handle inference requests & returns inference results

        Method Input
        =============
        x : List of Pillow images subject to required inference

        Method Output
        ==============
        MMOCR Results
        """
        x = np.stack([np.asarray(i) for i in x])
        response = self.stub.inference(self.input_processor(x))
        return self.output_processor(response)
    
    def __del__(self):
        """
        This method is used to close communication channel to GRPC server

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        self.channel.close()

if __name__ == '__main__':
    print(
        """
        Sample Usage:
        =============

        from tentacle import *

        server_ip = '0.0.0.0:4321'
        infer = mmocr_ocr(server_ip)

        pil_images = [PIL_Image_1, PIL_Image_2, PIL_Image_3, PIL_Image_4, ...]
        result = infer(pil_images)
        """
    )