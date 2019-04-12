from PIL import Image
import os.path
import glob


def convert_png(pngfile, outdir, width=200, height=200):
    img = Image.open(pngfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        new_img.save(os.path.join(outdir, os.path.basename(pngfile)))
    except Exception as e:
        print(e)


for pngfile in glob.glob("/home/txl/catkin_ws/src/getdata/image/*.png"):
    convert_png(pngfile, "/home/txl/catkin_ws/src/getdata/image1")