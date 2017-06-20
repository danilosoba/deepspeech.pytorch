import os
from shutil import copyfile

sourceDir = "101_ObjectCategories"
destinationDir = "caltech101"

if not os.path.exists(os.path.join(destinationDir, "train")):
    os.makedirs(os.path.join(destinationDir, "train"))

if not os.path.exists(os.path.join(destinationDir, "test")):
    os.makedirs(os.path.join(destinationDir, "test"))

for dirName, subdirList, fileList in os.walk(sourceDir):
    print('\nFOUND DIRECTORY: ', dirName)
    for fname in fileList:
        if int(fname[6:10]) <= 30:
            print('\tTRAIN: ', fname)
            print(os.path.join(dirName, fname))
            processingDir = os.path.join(destinationDir, "train", dirName[len(sourceDir)+1:])
            print(processingDir)
            if not os.path.exists(processingDir):
                os.makedirs(processingDir)
            print(os.path.join(processingDir, fname))
            copyfile(os.path.join(dirName, fname),
                     os.path.join(processingDir, fname))
        else:
            print('\tTEST:  ', fname)
            print(os.path.join(dirName, fname))
            processingDir = os.path.join(destinationDir, "test", dirName[len(sourceDir)+1:])
            print(processingDir)
            if not os.path.exists(processingDir):
                os.makedirs(processingDir)
            print(os.path.join(processingDir, fname))
            copyfile(os.path.join(dirName, fname),
                     os.path.join(processingDir, fname))

