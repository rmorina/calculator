import numpy as np
import scipy
import scipy.misc
import os
from PIL import Image
from PIL import ImageFilter

def readFile(path):
    with open(path, "rt") as f:
        return f.read()

def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

rows = 15
cols = 12
pics_per_symbol = 4
margin = 40
k = 1

for i in xrange(pics_per_symbol):
    path = 'symbols_data/+_%d.jpg' % (i+1)
    im_list = np.asarray(Image.open(path))
    im_list.setflags(write=True)
    (height, width) = im_list.shape
    cell_width = width/cols
    cell_height = height/rows
    for row in xrange(rows):
        for col in xrange(cols):

            top = row*cell_height + margin
            left = col*cell_width + margin
            right = left + cell_width - 2*margin
            bottom = top + cell_height - 2*margin

            slice_list = im_list[top:bottom, left:right]

            # some processing
            high_values_indices = slice_list > 180
            slice_list[high_values_indices] = 255
            rescaled_im_list = scipy.misc.imresize(slice_list, (28, 28), 'cubic')
            slice_im = Image.fromarray(rescaled_im_list)
            im_sharp = slice_im.filter(ImageFilter.SHARPEN)
            im_sharp = im_sharp.filter(ImageFilter.SHARPEN)

            # save the new processed file
            file_name = 'symbols_data/+/%d.png' % k
            im_sharp.save(file_name)
            k += 1