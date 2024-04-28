import h5py
import PIL.Image as pImg
import numpy as np
import glob
import os

def rgb2gray(img):
    return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.

def setTrianData(imgPath, h5Path, scale=4, pSize=33, pStride=14):
    h5_file = h5py.File(h5Path, 'w')
    lrPatches, hrPatches = [], []       
    for p in sorted(glob.glob(f'{imgPath}/*')):
        hr = pImg.open(p).convert('RGB')
        lrWidth, lrHeight = hr.width // scale, hr.height // scale
        width, height = lrWidth*scale, lrHeight*scale
        hr = hr.resize((width, height), resample=pImg.BICUBIC)
        lr = hr.resize((lrWidth, lrHeight), resample=pImg.BICUBIC)
        lr = lr.resize((width, height), resample=pImg.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = rgb2gray(hr)
        lr = rgb2gray(lr)
        for i in range(0, height - pSize + 1, pStride):
            for j in range(0, width - pSize + 1, pStride):
                lrPatches.append(lr[i:i + pSize, j:j + pSize])
                hrPatches.append(hr[i:i + pSize, j:j + pSize])
    h5_file.create_dataset('lr', data=np.array(lrPatches))
    h5_file.create_dataset('hr', data=np.array(hrPatches))
    h5_file.close()


def main():
    imgpath = "DIV2K/DIV2K_valid_256/"
    h5path = "DIV2K"
    if not os.path.exists(h5path):
        os.mkdir(h5path)
    setTrianData(imgpath, "DIV2K/mytest256.h5")
    print("complete")


if __name__ == '__main__':
    main()