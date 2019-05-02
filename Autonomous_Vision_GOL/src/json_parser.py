import json
import os
import glob
import time

json_path = "D:\\AndreiMihalcea\\berkley dataset\\labels\\100k\\train"
json_files = [f for f in glob.glob(json_path + "/*.json")]
classes_dict = dict()
# json_data = open(json_files[0])
# print(json_files[0])
# data = json.load(json_data)
# objects_data = data["frames"][0]["objects"]
# for object in  objects_data:
#     print(object["category"])

for file in json_files:
    json_data = open(file)
    data = json.load(json_data)
    objects_data = data["frames"][0]["objects"]
    for object in objects_data:
        if object["category"] == "train":
            print(file)
        # if not object["category"] in classes_dict.keys():
        #     classes_dict[object["category"]] = 1
        # else:
        #     classes_dict[object["category"]] += 1
print(classes_dict)
