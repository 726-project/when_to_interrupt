import sys
import numpy as np
import pandas as pd
import os
import math
import codecs, json
from glob import glob

SEP = '/'
PROCESSED_DIR = f'.{SEP}data{SEP}processed{SEP}*'
COMBINE_DIR = f'.{SEP}data{SEP}combined_jsons{SEP}*'
RAW_DIR = f'.{SEP}data{SEP}raw{SEP}*'
# HAT_ORIENTATION = 'vicon_hat_3_hat_3_orientation.csv'
# HAT_TRANSLATION = 'vicon_hat_3_hat_3_translation.csv'
IMAGE_RAW = 'img_raw_03.csv'
JSON_DATASET = f'.{SEP}dataset{SEP}json'
CSV_DATASET = f'.{SEP}dataset{SEP}csv'
CSV_FLAG = True
JSON_FLAG = True

if not os.path.exists(f'.{SEP}dataset'):
    os.mkdir(f'.{SEP}dataset')
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
if not os.path.exists(JSON_DATASET+SEP+'openpose'):
    os.mkdir(JSON_DATASET+SEP+'openpose')
if not os.path.exists(CSV_DATASET+SEP+'openpose'):
    os.mkdir(CSV_DATASET+SEP+'openpose')

def writeToJson(hat, openpose, combined, file_date, writeFlag):
    if not writeFlag: return
    else:
        hat_list = hat.values.tolist()
        openpose_combined = combined.values.tolist()
        openpose_json = openpose.values.tolist()

        json.dump(openpose_json, codecs.open(JSON_DATASET + SEP + 'openpose' + SEP + file_date+'_openpose.json', 'w', encoding='utf-8'), \
        separators=(',', ':'), sort_keys=True, indent=2)

        json.dump(openpose_combined, codecs.open(JSON_DATASET + SEP + 'all' + SEP + file_date+'_all.json', 'w', encoding='utf-8'), \
        separators=(',', ':'), sort_keys=True, indent=2)

        json.dump(hat_list, codecs.open(JSON_DATASET + SEP + 'hat' + SEP + file_date+'_hat.json', 'w', encoding='utf-8'), \
        separators=(',', ':'), sort_keys=True, indent=2)

def writeToCSV(hat, openpose, combined, file_date, writeFlag):
    if not writeFlag: return
    else:
        hat.iloc[:,2:].to_csv(CSV_DATASET + SEP + 'hat' + SEP + file_date + '_hat.csv')
        combined.to_csv(CSV_DATASET + SEP + 'all' + SEP + file_date + '_all.csv')
        openpose.to_csv(CSV_DATASET + SEP + 'openpose' + SEP + file_date + '_openpose.csv')

def getHatFiles(hat_dir):
    allFiles = glob(hat_dir + '*')
    res = [i.split(SEP)[-1] for i in allFiles if i.split(SEP)[-1].startswith('vicon_hat')]
    if len(res) >= 2:
        hat_orientation = [i for i in res if i.split('_')[-1].startswith('orientation')]
        hat_translation = [i for i in res if i.split('_')[-1].startswith('translation')]
        return hat_orientation[-1], hat_translation[-1]
    else:
        print("There are less than 2 hat orientation csv files")

def readAndProcess(file_date):
    hat_dir = f'.{SEP}data{SEP}raw{SEP}{file_date}.bag{SEP}'
    json_file_path = COMBINE_DIR[:-1] + file_date + 'bag_' + IMAGE_RAW[:-4] + '_combined' + '.json'
    hat_orie, hat_trans = getHatFiles(hat_dir)
    hat_orientation = pd.read_csv(hat_dir + hat_orie,sep=',',names=orien_cols)
    # print(json_file_path)
    hat_translation = pd.read_csv(hat_dir + hat_trans,sep=',',names=trans_cols)
    img_raw = pd.read_csv(hat_dir + IMAGE_RAW,sep=',',names=img_cols)
    img_raw['id'] = img_raw['id'].astype(int).astype(str)
    hat_orientation['id'] = hat_orientation['id'].apply(math.trunc).astype(int).astype(str)
    hat_translation['id'] = hat_translation['id'].apply(math.trunc).astype(int).astype(str)
    hat_orie_mean = hat_orientation.groupby(['id']).agg('mean')
    hat_trans_mean = hat_translation.groupby(['id']).agg('mean')
    # print(hat_orie_mean)
    hat_merge = pd.merge(hat_orie_mean, hat_trans_mean,on='id').reset_index().sort_values(by=['id']) # Reset index here to give index to df
    hat_final = pd.merge(img_raw, hat_merge,on='id').sort_values(by=['id'])
    # print(hat_final)
    openpose = pd.read_json(json_file_path)
    # openpose[0].astype(str)
    # openpose.loc[openpose[0] == 1, 0] = 'Not Confused'
    # openpose.loc[openpose[0] == 2, 0] = 'Uncertain'
    # openpose.loc[openpose[0] == 3, 0] = 'Confused'
    openpose[0] = openpose[0] - 1
    # print(openpose)
    combined_json =pd.merge(openpose,hat_final.iloc[:,2:],left_index=True,right_index=True)
    combined_json = combined_json[combined_json[0] != -1]  #filter out objects with 0 as a label
    # print(combined_json[combined_json.columns[-6:combined_json.columns.size]])
    hat_toWrite = pd.concat([combined_json.iloc[:,0], \
        combined_json.iloc[:,-6:combined_json.columns.size]],axis = 1)
    # print(pd.unique(combined_json.iloc[:,0:55][0]))
    # print(combined_json.iloc[:,0:55])
    # print(openpose[openpose[0] != -1])
    hat_toWrite.dropna()
    combined_json.dropna()
    writeToJson(hat_toWrite, combined_json.iloc[:,0:55], combined_json, file_date, JSON_FLAG)
    writeToCSV(hat_toWrite, combined_json.iloc[:,0:55], combined_json, file_date, CSV_FLAG)

orien_cols = ['id','roll','pitch','yaw']
trans_cols = ['id','x','y','z']
img_cols = ['time','id']
json_list = glob(COMBINE_DIR)
hat_csv_list = glob(RAW_DIR)
csv_name_list = []
for i in hat_csv_list:
    csv_name_list.append(i.split(SEP)[-1][:-4])
# print(csv_name_list)
for name in json_list:
    dateOfDir = name.split(SEP)[-1].strip()[0:19]
    # print(dateOfDir)
    if dateOfDir in csv_name_list:
        print("Processing file: " + str(dateOfDir) +".bag")
        readAndProcess(dateOfDir)
    else:
        print("File " + str(dateOfDir) + " Not Found!")
