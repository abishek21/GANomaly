import os
import glob



#
# path_train = 'UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/'
# #rootdir = os.getcwd()
# filenames=[]
# for subdir, dirs, files in os.walk(path_train):
#     for file in files:
#         #print os.path.join(subdir, file)
#         filepath = subdir + os.sep + file
#
#         if filepath.endswith(".tif"):
#             filenames.append(filepath)
# print(filenames[:10])
# train_dir="train"
# if not os.path.exists(train_dir):
#         os.makedirs(train_dir)
#
# for i in range(len(filenames)):
#     new_name=train_dir+"\\"+str(i)+".png"
#     os.rename(filenames[i],new_name)

path_test = 'UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/'

filenames_test=[]
for subdir, dirs, files in os.walk(path_test):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".tif"):
            filenames_test.append(filepath)
print(filenames_test[:10])
test_dir="test"
if not os.path.exists(test_dir):
        os.makedirs(test_dir)

for i in range(len(filenames_test)):
    new_name=test_dir+"\\"+str(i)+".png"
    os.rename(filenames_test[i],new_name)