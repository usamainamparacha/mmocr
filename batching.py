import os
from PIL import Image
import pandas as pd
from tentacle import *
from utils import connect_to_mongo, push_to_mongo
from ast import literal_eval


TEST_IMG_DIR = '/home/haris/test_frames'


def merge_dictionary_list(dict_list):
  return {
    k: [d.get(k) for d in dict_list if k in d] 
    for k in set().union(*dict_list)}


def unpack_cols(result):
  df = pd.DataFrame(result)
  # df['result'] = df['result'].apply(lambda x: literal_eval(x)) #COMMENT THIS OUT when using in deployment. This is ONLY required when using a test csv
  df_merged_dicts = df['result'].apply(lambda x: merge_dictionary_list(x))
  frame_inferlist = df_merged_dicts.tolist()
  fdf = merge_dictionary_list(frame_inferlist)
  df_final = pd.DataFrame(fdf)
  return df_final



if __name__ == '__main__':
    server_ip = '10.10.56.184:4321'
    infer = mmocr_ocr(server_ip)
    pil_img_list = list()
    os.chdir(TEST_IMG_DIR)
    for i in os.listdir(os.getcwd()):
        pil_img_list.append(Image.open(i).resize((852, 480)))
    result = infer(pil_img_list)
    final = unpack_cols(result)
    client = connect_to_mongo('10.10.56.115:27017')
    push_to_mongo(client, final, 'OLTP', 'mmocr')

    print(-1)