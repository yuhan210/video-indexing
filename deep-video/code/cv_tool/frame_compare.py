from PIL import Image
import imagehash


def phash(a, b):
    a_hash = imagehash.phash(a)
    b_hash = imagehash.phash(b)

    return a_hash - b_hash

def dhash(a, b):
    a_hash = imagehash.dhash(a)
    b_hash = imagehash.dhash(b)

    return a_hash - b_hash

def ahash(a, b):
    a_hash = imagehash.average_hash(a)
    b_hash = imagehash.average_hash(b)
    
    return a_hash - b_hash





'''
Return the hamming distance. Larger than 5 is different
'''

if __name__ == "__main__":

    a_path = "/home/t-yuche/caffe-install/opencv-3.0.0/samples/data/lena.jpg"
    b_path = "/home/t-yuche/caffe-install/opencv-3.0.0/samples/data/lena_tmpl.jpg"
    img_a = Image.open(a_path)
    img_b = Image.open(b_path)
    print dhash(img_a, img_b)

