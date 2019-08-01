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
import matplotlib.pyplot as plt
import matplotlib as mpl

ia.seed(4) # set seed for consistent cropping

# CONSTANTS  ----------------

chars = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

# each char has a different height and width
chars_set = set(chars)
tall_chars = set('1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZbdfhijklt')
low_chars = set('gpqy')
reg_chars = chars_set - tall_chars - low_chars

# make font_imgs root
root = './font_imgs'
if not os.path.exists(root):
    os.mkdir(root)

# image creation constant
width = 64
height = 64
# fixed_height_offset = -10  # correction for height centering on image (move up)

# more negative moves char up
tall_height_offset = -8  # -8 is good
low_height_offset = -11  # -11 is good
reg_height_offset = -15  # -15 is good

# set font sizes for each char type
low_font_size = 55  # make same as reg font size
reg_font_size = 55
tall_font_size = 50

# vary the translation and font size here
font_size_range = (8, 8)

tall_height_offset_range = (0, 3)  # moving down randomly only is better
reg_height_offset_range = (0, 3)

width_offset_range = (-6, 6)

N = 10 # num copies of each char


# end constants ---------------



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
    color = list(np.random.choice(range(0, 150), size=1))
    color = tuple([color, color, color])  # convert to tuple
    return color

def random_font_color(background_color_num):
    # take in background color to ensure a min contrast

    # contrast minimum between background
    margin = 20
    rand_num = list(np.random.choice(range(0 + margin, 150), size=1))

    pos_or_neg = random.choice([-1,1])  # decides to add or subtract

    font_color_num = pos_or_neg * rand_num + background_color_num

    # if it's negative, flip the sign
    if font_color_num < 0:
        font_color_num = -1 * pos_or_neg * rand_num + background_color_num

    font_color_num = tuple([font_color_num, font_color_num, font_color_num])  # convert to tuple
    return font_color_num

def random_offset(offset_range):
    return random.randint(offset_range[0], offset_range[1])

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

def save_image(img_path, gray_img_path):
    # Open and convert to grayscale here
    I = Image.open(img_path).convert('L')  # ensure a gray image
    I = np.asarray(I)  # convert to numpy
    # I = imread(fn)  # alternate between this and PIL image open

    I = np.invert(I)
    I = I / 255
    I = I.astype('float32')
    # convert to numpy

    _, ax = plt.subplots(1, 2)
    ax[0].imshow(I, cmap=mpl.cm.bone)
    plt.savefig(gray_img_path)
    plt.close('all')


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

                font_size = None
                height_offset = None
                height_offset_range = None

                # correct height offset for char height
                if char in tall_chars:
                    height_offset = tall_height_offset
                    font_size = tall_font_size
                    height_offset_range = tall_height_offset_range
                elif char in low_chars:
                    height_offset = low_height_offset
                    font_size = low_font_size
                    height_offset_range = reg_height_offset_range
                else:
                    height_offset = reg_height_offset
                    font_size = reg_font_size
                    height_offset_range = reg_height_offset_range

                # select font file and create font object, and set size
                font_obj = ImageFont.truetype(font_path, font_size + random_offset(font_size_range))

                background_color = random_back_color()

                # create an image as rgb
                img = Image.new("L", (width, height), 5)

                # creates a draw object for im, to allow you to write on the image
                draw_object = ImageDraw.Draw(img)

                text_width, text_height = draw_object.textsize(char, font=font_obj)

                width_pos = (width-text_width)/2 + random_offset(width_offset_range)
                height_pos = (height-text_height)/2 + random_offset(height_offset_range) + height_offset

                # tries to center char in the image
                draw_object.text((width_pos, height_pos), char, align='center', font=font_obj, fill=random_font_color(background_color))
                img_np = np.asarray(img)  # need to convert image to numpy array

                aug_img = seq.augment_image(img_np)  # apply augmentation to image

                # save image
                img_name = '{}.png'.format(str(i).zfill(3))  # add padding to name and same path name
                img_path = os.path.join(char_root, img_name)
                img = Image.fromarray(aug_img)  # convert back to PIL
                
                img.save(img_path)  # save as PIL


                # # save a gray img too normalized too, for testing
                # gray_img_name = '{}_gray.png'.format(str(i).zfill(3))
                # gray_img_path = os.path.join(char_root, gray_img_name)
                # save_image(img_path, gray_img_path)




if __name__ == "__main__":
    # execute only if run as a script
    generate()
