import numpy as np
from PIL import Image
import os

def readFiles(tpath):
    txtLists = os.listdir(tpath)
    List = []
    for t in txtLists:
        t = tpath + "/" + t
        List.append(t)
    return List

store = 'D:\Docs\programs\OracleRecognition\Low_level_CV_PPL/data/Dataset2-npy/target/'
target = 'D:\Docs\programs\OracleRecognition\Low_level_CV_PPL/data/Dataset2/target/'
targets = readFiles(target)

for t in targets:
    name = t.split('/')[-1].split('.')[0]
    img = Image.open(t).convert('RGB')
    s = store + name + '.npy'
    np.save(s, np.array(img))
    # pic = Image.fromarray(s)
    # pic.show()
