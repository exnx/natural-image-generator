# -*- coding: utf-8 -*-

import os
from PIL import Image, ImageFont, ImageDraw
import sys
import random
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import random
import numpy as np
import random

ia.seed(4) # set seed for consistent cropping

chars = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
tall_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZbdfhijklt')

# make font_imgs root
root = './font_imgs'
if not os.path.exists(root):
    os.mkdir(root)

# image creation constant
width = 64
height = 64
fixed_height_offset = -10  # correction for height centering on image (move up)
default_small_font_size = 50
default_tall_font_size = 65
font_size_range = 5
translation_offset_range = 3

N = 100 # num copies of each char

def get_fonts_list(dir_path):

    font_names_list = []
    font_files_list = []

    print('dir path', dir_path)

    for file in os.listdir(dir_path):
        if file.endswith(".ttf"):

            font_file_name = os.path.basename(file)
            font_name = font_file_name.split('.')[0]

            font_names_list.append(font_name)
            font_files_list.append(font_file_name)

    return font_names_list, font_files_list

def random_back_color():
    color = list(np.random.choice(range(210,256), size=3))
    color = tuple(color)  # convert to tuple
    return color

def random_font_color():
    color = list(np.random.choice(range(0,50), size=3))
    color = tuple(color)  # convert to tuple
    return color

def random_offset(num):
    return random.randint(-num, num)

def get_aug_obj():
    # # def augmentation sequence
    seq = iaa.Sequential(
        [
            iaa.SomeOf((0, None), [
                iaa.AdditiveGaussianNoise(loc=0, scale=0.02*255, per_channel=0.5),  # noise
                iaa.Add((-20, 20), per_channel=0.5),  # random add pixel values
                iaa.Sharpen(alpha=(0,0.3), lightness=(0.75, 1.5)),  # sharpen
                iaa.Multiply((0.5, 1.5), per_channel=0.5),  # increase intensity
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # contrast
                iaa.OneOf([
                    iaa.GaussianBlur((0, 0.5)),  # choose one type of blur
                    iaa.AverageBlur(k=(1, 1)),
                    iaa.MedianBlur(k=(1,1)),
                ])
            ]),
        ],
        # do all of the above augmentations in random order
        random_order=True
    )

    return seq

def generate():

    fonts_root_dir = 'fonts'

    # retrieve list of font names and font files
    font_names_list, font_files_list = get_fonts_list(fonts_root_dir)

    # generate the augmentation object
    seq = get_aug_obj()

    # loop thru fonts
    for font_idx, font_name in enumerate(font_names_list):

        print('creating images for font: {}'.format(font_name))

        # make font root dir
        font_root = './{}/{}'.format(root, font_name)
        if not os.path.exists(font_root):
            os.mkdir(font_root)

        # loop thru each char
        for char in chars:
            char_dir = ''
            # check if is capital char
            if char.isupper():
                char_dir = char + char  # need to differentiate upper case letter dir
            else:
                char_dir = char

            # make char root dir
            char_root = './{}/{}/{}'.format(root, font_name, char_dir)
            if not os.path.exists(char_root):
                os.mkdir(char_root)

            # make N copies of the char
            for i in range(N):

                font_path = fonts_root_dir + '/' + font_files_list[font_idx]

                # select font file and create font object, and set size
                font_obj_small = ImageFont.truetype(font_path, default_small_font_size + random_offset(font_size_range))
                font_obj_tall = ImageFont.truetype(font_path, default_tall_font_size + random_offset(font_size_range))

                # create an image as rgb
                img = Image.new("RGB", (width, height), random_back_color())

                # creates a draw object for im, to allow you to write on the image
                draw_object = ImageDraw.Draw(img)

                # if it's a tall char, use the small font (to fit inside image size)
                if char in tall_chars:
                    font_obj = font_obj_small
                else:
                    font_obj = font_obj_tall

                text_width, text_height = draw_object.textsize(char, font=font_obj)

                # tries to center char in the image
                draw_object.text(((width-text_width + random_offset(translation_offset_range))/2,(height-text_height + random_offset(translation_offset_range))/2 + \
                    fixed_height_offset + random_offset(translation_offset_range)), char, align='center', font=font_obj, fill=random_font_color())

                img_np = np.asarray(img)  # need to convert image to numpy array
                aug_img = seq.augment_image(img_np)  # apply augmentation to image
                img_name = '{}.png'.format(str(i).zfill(3))  # add padding to name and same path name

                # save image
                img_path = os.path.join(char_root, img_name)
                img = Image.fromarray(aug_img)  # convert back to PIL
                img.save(img_path)  # save as PIL

if __name__ == "__main__":
    # execute only if run as a script
    generate()
