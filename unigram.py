# -*- coding: utf-8 -*-

import os
from PIL import Image, ImageFont, ImageDraw
import sys
import random


root = './font_imgs'
if not os.path.exists(root):
    os.mkdir(root)

# 1. loop thru each font
# 2. loop through each char
# 3. create image, save


# seed = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+=-'
chars = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
font_list_names = ['arial.ttf', 'vera.ttf', 'arialbold.ttf', 'arialitalic.ttf']
# font_names = ['arial']
font_names = ['arial', 'vera', 'arialbold', 'arialitalic']

# store font file name in a dictionary, key=font_name, value=font_filename
font_filename = {}

for index, font_name in enumerate(font_names):
    font_filename[font_name] = font_list_names[index]


width = 128
height = 64
factor = 1  # select a scaling factor for sizing the image/letters



for font in font_names:

    img_index = 0  # reset for each font

    # create a new font dir for images
    root = './font_imgs/{}'.format(font)
    if not os.path.exists(root):
        os.mkdir(root)

    # # repeat 10 times
    # for i in range(10):

    for char in chars:

        # select font file and create font object
        font_obj = ImageFont.truetype(font_filename[font], 40 * factor)
    
        # create an image as rgb
        im = Image.new("RGB", (width * factor, height * factor), (255, 255, 255))

        # creates a draw object for im, to allow you to write on the image
        draw_object = ImageDraw.Draw(im)

        # draw the text with the new font on the changed img
        draw_object.text((40 * factor + 8, 10 * factor), char, font=font_obj, fill="#000000")

        img_index += 1  # increment img index

        # add padding to name and same path name
        img_name = '{}_{}.png'.format(font, str(img_index).zfill(3))

        # save image
        img_path = os.path.join(root, img_name)

        im.save(img_path)












