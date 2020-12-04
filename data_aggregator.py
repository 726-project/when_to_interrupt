import sys
import numpy as np
import pandas as pd
import os
import math
import codecs, json
from glob import glob

SEP = '/'
PROCESSED_DIR = f'.{SEP}data{SEP}processed{SEP}*'
RAW_DIR = f'.{SEP}data{SEP}raw{SEP}*'
HAT_ORIENTATION = 'vicon_hat_3_hat_3_orientation.csv'
HAT_TRANSLATION = 'vicon_hat_3_hat_3_translation.csv'
IMAGE_RAW = 'img_raw_03.csv'
JSON_DATASET = f'.{SEP}dataset{SEP}json'
CSV_DATASET = f'.{SEP}dataset{SEP}csv'
CSV_FLAG = True
JSON_FLAG = True

if not os.path.exists(JSON_DATASET):
    os.mkdir(JSON_DATASET)
if not os.path.exists(CSV_DATASET):
    os.mkdir(CSV_DATASET)
if not os.path.exists(JSON_DATASET+SEP+'hat'):
    os.mkdir(JSON_DATASET+SEP+'hat')
if not os.path.exists(CSV_DATASET+SEP+'hat'):
    os.mkdir(CSV_DATASET+SEP+'hat')
if not os.path.exists(JSON_DATASET+SEP+'all'):
    os.mkdir(JSON_DATASET+SEP+'all')
if not os.path.exists(CSV_DATASET+SEP+'all'):
    os.mkdir(CSV_DATASET+SEP+'all')

def writeToJson(hat, combined, file_date, writeFlag):
    if not writeFlag: return
    else:
        hat_list = hat.iloc[:,2:].values.tolist()
        openpose_json = combined.values.tolist()
        json.dump(openpose_json, codecs.open(JSON_DATASET + SEP + 'all' + SEP + file_date+'_all.json', 'w', encoding='utf-8'), \
        separators=(',', ':'), sort_keys=True, indent=2)

        json.dump(hat_list, codecs.open(JSON_DATASET + SEP + 'hat' + SEP + file_date+'_hat.json', 'w', encoding='utf-8'), \
        separators=(',', ':'), sort_keys=True, indent=2)

def writeToCSV(hat, combined, file_date, writeFlag):
    if not writeFlag: return
    else:
        hat.iloc[:,2:].to_csv(CSV_DATASET + SEP + 'hat' + SEP + file_date + '_hat.csv')
        combined.to_csv(CSV_DATASET + SEP + 'all' + SEP + file_date + '_all.csv')

def readAndProcess(file_date):
    hat_dir = f'.{SEP}data{SEP}raw{SEP}{file_date}.bag{SEP}'
    json_file_path = PROCESSED_DIR[:-1] + 'raw_' + file_date + 'bag_' + IMAGE_RAW[:-4] + '_combined.json'
    hat_orientation = pd.read_csv(hat_dir + HAT_ORIENTATION,sep=',',names=orien_cols)
    hat_translation = pd.read_csv(hat_dir + HAT_TRANSLATION,sep=',',names=trans_cols)
    img_raw = pd.read_csv(hat_dir + IMAGE_RAW,sep=',',names=img_cols)
    img_raw['id'] = img_raw['id'].astype(int).astype(str)
    hat_orientation['id'] = hat_orientation['id'].apply(math.trunc).astype(int).astype(str)
    hat_translation['id'] = hat_translation['id'].apply(math.trunc).astype(int).astype(str)
    hat_orie_mean = hat_orientation.groupby(['id']).agg('mean')
    hat_trans_mean = hat_translation.groupby(['id']).agg('mean')
    hat_merge = pd.merge(hat_orie_mean, hat_trans_mean,on='id').reset_index().sort_values(by=['id']) # Reset index here to give index to df
    hat_final = pd.merge(img_raw, hat_merge,on='id').sort_values(by=['id'])
    openpose = pd.read_json(json_file_path)
    combined_json =pd.merge(openpose,hat_final.iloc[:,2:],left_index=True,right_index=True)
    combined_json = combined_json[combined_json[0] != 0]  #filter out objects with 0 as a label
    writeToJson(hat_final, combined_json, file_date, JSON_FLAG)
    writeToCSV(hat_final, combined_json, file_date, CSV_FLAG)

orien_cols = ['id','roll','pitch','yaw']
trans_cols = ['id','x','y','z']
img_cols = ['time','id']
json_list = glob(PROCESSED_DIR)
hat_csv_list = glob(RAW_DIR)
json_name_list = []
for i in json_list:
    json_name_list.append(i.split(SEP)[-1].strip()[4:23])
print(json_name_list)
for name in hat_csv_list:
    dateOfDir = name.split(SEP)[-1][:-4]
    if dateOfDir in json_name_list:
        print("Processing file: " + str(dateOfDir) +".bag")
        readAndProcess(dateOfDir)
    else:
        print("File " + str(dateOfDir) + " Not Found!")
