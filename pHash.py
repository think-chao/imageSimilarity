import imagehash
import os
from PIL import Image

if __name__ == '__main__':
    image_path = 'C:/Users/v_wangchao3/Pictures/Camera Roll'
    image_list = os.listdir(image_path)
    for img in image_list:
        hash1 = imagehash.dhash(Image.open(r'C:\Users\v_wangchao3\Pictures\Camera Roll\6.jpg'))
        hash2 = imagehash.dhash(Image.open(image_path+'/'+img))
        out_score = hash1 - hash2
        print(img)
        if 0 <= out_score < 30:

            print("汉明距离为%d,哇呜这两幅图片很像" % (out_score))
            Image.open(image_path+'/'+img).show()
        elif 5 <= out_score < 40:
            print("汉明距离为%d,这两幅图片很像,但又有一些差别" % (out_score))
        else:
            print("汉明距离为%d,这两幅图片好像差距挺大的" % (out_score))








