"""
Library for segmenting image into separate digits.
"""

from PIL import Image, ImageOps
import numpy

BINARY_THRESHOLD = 200
CUSHION = 20

def scale(img):
    """
    Converts to binary image of background and foreground.
    """
    result = [[1 if x > BINARY_THRESHOLD else 0 for x in row] for row in img]
    return result

def get_connected_components(img):
    """
    Returns list of connected components, where each connected
    component, is a list of pixels (represented as (row,col).
    """
    already_labeled = []
    connected_components = []
    pixel_queue = []

    def get_fg_neighbors(row,col):
        neighbors = []
        min_row = max(row-1,0)
        max_row = min(row+1, len(img)-1)
        min_col = max(col-1,0)
        max_col = min(col+1, len(img[0])-1)
        for r in xrange(min_row, max_row+1):
            for c in xrange(min_col, max_col+1):
                if img[r][c] == 1 and (r,c) not in already_labeled:
                    neighbors.append((r,c))
        return neighbors

    for row in xrange(len(img)):
        for col in xrange(len(img[0])):
            if img[row][col] == 1 and (row,col) not in already_labeled:
                already_labeled.append((row,col))
                cc = []
                cc.append((row,col))
                pixel_queue.append((row,col))
                while len(pixel_queue) > 0:
                    cur_pixel = pixel_queue[0]
                    pixel_queue = pixel_queue[1:]
                    neighbors = get_fg_neighbors(cur_pixel[0], cur_pixel[1])
                    already_labeled += neighbors
                    cc += neighbors
                    pixel_queue += neighbors
                connected_components.append(cc)
    return connected_components


def get_image_segments(connected_components):
    """
    Returns segments of original images according to connected
    components.
    """
    image_segments = []
    for cc in connected_components:
        min_row = cc[0][0]
        min_col = cc[0][1]
        max_row = cc[0][0]
        max_col = cc[0][1]
        for pixel in cc:
            if pixel[0] < min_row:
                min_row = pixel[0]
            if pixel[0] > max_row:
                max_row = pixel[0]
            if pixel[1] < min_col:
                min_col = pixel[1]
            if pixel[1] > max_col:
                max_col = pixel[1]
        height = max_row - min_row + CUSHION*2
        width = max_col - min_col + CUSHION*2
        top_left_row = max(min_row - CUSHION, 0)
        top_left_col = max(min_col - CUSHION, 0)
        image_segments.append((top_left_row, top_left_col, width, height))
    return image_segments


def get_segments(path):
    img = Image.open(path).convert('L')
    img = ImageOps.invert(img)
    img_list = numpy.asarray(img)
    scaled_img = scale(img_list)
    connected_components = get_connected_components(scaled_img)
    image_segments = get_image_segments(connected_components)
    images = []
    for seg in image_segments:
        img_seg = []
        top_left_row, top_left_col, width, height = seg
        for r in xrange(top_left_row, top_left_row + height):
            img_seg.append(img_list[r][top_left_col:top_left_col+width])
        images.append(img_seg)
    resized_images = []
    for i in images:
        resized_image = Image.fromarray(numpy.array(i), 'L').resize((28,28))
        resized_array = numpy.asarray(resized_image)
        resized_array = resized_array / 255.0
        resized_images.append(resized_image)
    return resized_images
