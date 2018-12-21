import re
import csv
import numpy as np
import pandas as pd
import pickle

valid_length_percentage = 0.05

original_food_feature_path = "../data/original_data/train.csv"
df = pd.read_csv(original_food_feature_path, sep=',', names=['source', 'target'], header=None)#source,target
print("df[source]")
print(df["source"])
print("df[target]")
print(df["target"])
print("df[source][0]: ", df["source"][0])
print("df[target][0]: ", df["target"][0])
print("df[source].shape[0]: ", df["source"].shape[0])
all_length = df["source"].shape[0]
valid_length = int(all_length * valid_length_percentage)

valid_source = []
valid_target = []
train_source = []
train_target = []
for i in range(df["source"].shape[0]):
    source = df["source"][i]
    target = df["target"][i]
    if(i < valid_length):
        valid_source.append(source)
        valid_target.append(target)
    else:
        train_source.append(source)
        train_target.append(target)

output_path = "../data/split_data/valid_data.csv"#date,userid,foodid
with open(output_path, 'w') as f:
    #f.write("date,userid,foodid\n")
    for i in range(len(valid_source)):
        source = valid_source[i]
        target = valid_target[i]
        f.write('%s,%s\n' % (source, target))


output_path = "../data/split_data/train_data.csv"#date,userid,foodid
with open(output_path, 'w') as f:
    #f.write("date,userid,foodid\n")
    for i in range(len(train_source)):
        source = train_source[i]
        target = train_target[i]
        f.write('%s,%s\n' % (source, target))


#valid_source = df["source"][:valid_length]
#valid_target = df["target"][:valid_length]
#train_source = df["source"][valid_length:all_length]
#train_target = df["target"][valid_length:all_length]

"""
print("type(valid_source): ", type(valid_source))
valid_data = pd.concat([valid_source, valid_target], axis=1, sort=False)
print("valid_data[source][0]: ", valid_data["source"][0])
print("valid_data[target][0]: ", valid_data["target"][0])
train_data = pd.concat([train_source, train_target], axis=1, sort=False)
print("train_data[source][valid_length]: ", train_data["source"][valid_length])
print("train_data[target][valid_length]: ", train_data["target"][valid_length])
#print("train_data[source][0]: ", train_data["source"][0])
#print("train_data[target][0]: ", train_data["target"][0])

valid_food_feature_path = "../data/split_data/valid_data.csv"
df.to_csv(valid_food_feature_path, sep=',')

train_food_feature_path = "../data/split_data/train_data.csv"
df.to_csv(train_food_feature_path, sep=',')
"""