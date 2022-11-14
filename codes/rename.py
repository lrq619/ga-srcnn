import os
from config import GAME

count = 1
workPATH='pictures/%s/720P'%GAME
for file in os.listdir(workPATH):
    try:
        os.rename('%s/%s'%(workPATH,file), '%s/%d.png'%(workPATH,count))
    except:
        pass
    count += 1