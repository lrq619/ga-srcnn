from torchvision import transforms
import PIL.Image as pil_image
from config import GAME
import os

game_path = 'pictures/%s/720P'%GAME
ITRATION_TIME = len(os.listdir(game_path)) 




def reduce_resolution(img,downmag):
    w = img.width
    h = img.height
    down_size = (int(h/downmag),int(w/downmag))
    
    down = transforms.Resize(down_size)
    up = transforms.Resize((h,w))
    res = down(img)
    res = up(res)
    return res

for i in range(1,ITRATION_TIME+1):
    orig_file = 'pictures/%s/720P/%d.png'%(GAME,i)
    orig_img = pil_image.open(orig_file).convert('RGB')

    lowres_img = reduce_resolution(orig_img,2)
    lowres_img.save('pictures/%s/mylow/%d.png'%(GAME,i))
    