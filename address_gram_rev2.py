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


# image creation constant
width = 128
height = 128

num_lines = 2
num_grams = 100 # num copies of grams for a given font

chars = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ             '  # adding spaces to increase likelihood

# more negative moves char up
height_offset = -4  # -11 is good

# set font sizes for each char type
low_font_size = 55  # make same as reg font size
reg_font_size = 55
tall_font_size = 50

# vary the translation and font size here
font_size_range = (0, 0)
height_offset_range = (0, 0)  # moving down randomly only is better
width_offset_range = (0, 0)

line_length = 12  # num chars in line
font_size = 14  # font size of chars


# make font_imgs root
root = './address_gram_images'
if not os.path.exists(root):
    os.mkdir(root)


def get_margin_color(background_color, min_margin, top):

    margin = random.randint(min_margin, top)  # calc margin
    sign = random.choice([-1, 1])  # select sign (add or subtract)
    color = sign * margin + background_color # get the color

    # if color is out of bounds, flip the sign
    if color < 0 or color > 255:
        color = -1 * sign * margin + background_color
    return color

def alter_background_color(background_color):

    min_margin = 10
    top = 30

    # use margin color to slight alter within a range
    r = get_margin_color(background_color[0], min_margin, top)
    g = get_margin_color(background_color[1], min_margin, top)
    b = get_margin_color(background_color[2], min_margin, top)
    sign = random.choice([-1, 1])  # select sign (add or subtract)
    
    color = [r, g, b]
    return tuple(color)

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

def arc(draw_obj, color, bbox, line_weight):

    angle1 = random.randint(0, 360)
    angle2 = random.randint(0, 360)

    draw_obj.arc(bbox, start=min(angle1, angle2), end=max(angle1, angle2), fill=color, width=1)

def line(draw_obj, color, bbox, line_weight):
    draw_obj.line(bbox, fill=color, width=1)

def rectangle(draw_obj, color, bbox, line_weight):

    draw_obj.rectangle(bbox, fill=None, outline=color, width=1)
    # print('drew rectangle')

def random_back_color():
    color = list(np.random.choice(range(130, 255), size=3))
    color = tuple(color)  # convert to tuple
    return color

def get_bbox():
    x1 = random.randint(0, width)
    x2 = random.randint(0, width)

    y1 = random.randint(0, height)
    y2 = random.randint(0, height)

    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)) 

def draw_random_shape(draw_obj, background_color):

    color = alter_background_color(background_color)  # a tuple
    bbox = get_bbox()  # a list of 4 numbers
    line_weight = random.uniform(0.05, 0.4)

    options = {0:rectangle, 1:arc, 2:line}
    # options = {0:'rectangle', 1:'arc', 2:'circle', 3:'line', 4:'ellipse', 5:'pieslice'}
    
    rand_num = random.randint(0, len(options)-1)

    options[rand_num](draw_obj, color, bbox, line_weight)

def random_font_color(background_color):

    # font color selection depends on background color

    top = 100
    margin_min = 40
    gray_margin = 50

    margin_met = False

    while not margin_met:

        r = get_margin_color(background_color[0], margin_min, top)
        g = get_margin_color(background_color[1], margin_min, top)
        b = get_margin_color(background_color[2], margin_min, top)

        # need to check grayscale margin is met to have enough contrast 
        if abs ((r + g + b)/3 - sum(background_color) / len(background_color) ) > gray_margin:
            margin_met = True
        # else:
            # print('margin not met')

    color = [r, g, b]
    color = tuple(color)  # convert to tuple
    return color

def random_offset(offset_range):
    return random.randint(offset_range[0], offset_range[1])

def get_aug_obj():
    # # def augmentation sequence
    seq = iaa.Sequential(
        [

            iaa.Sometimes(0.7,
                iaa.OneOf([
                    iaa.GaussianBlur((0, 0.5)),  # choose one type of blur, larger std, more blur
                    iaa.AverageBlur(k=(1, 1)),
                    iaa.MedianBlur(k=(1,1)),
                ])
            ),

            iaa.SomeOf((1, None), [
                # iaa.Resize({"height": (0.9, 1), "width": (0.9, 1)}), # this might mess things up actually
                iaa.AdditiveGaussianNoise(loc=0, scale=0.01*255, per_channel=0.5),  # noise
                iaa.Add((-10, 10), per_channel=0.5),  # random add pixel values
                # iaa.Sharpen(alpha=(0,0.3), lightness=(0.85, 1.25)),  # this might mess it up too
                iaa.Multiply((0.8, 1.2), per_channel=0.5),  # increase intensity
                # iaa.ContrastNormalization((0.8, 1.25), per_channel=0.3),  # contrast
                iaa.Dropout(p=(0, 0.01), per_channel=0.4),
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

def write_lines(img, draw_object, font_obj, background_color):

    first_line = ''
    second_line = ''

    # create number of letters for two lines
    for i in range(line_length):

        #  choose a random char
        letter = random.choice(chars)
        first_line += letter

        letter = random.choice(chars)
        second_line += letter

    # get the height, width of the 2 lines
    text_width_first, text_height_first = draw_object.textsize(first_line, font=font_obj)
    text_width_second, text_height_second = draw_object.textsize(second_line, font=font_obj)

    # calc width and height of each line, and for spacing between
    text_width = max(text_width_first, text_width_second)  # use width of the widest line
    vert_space = int(max(text_height_first, text_height_second))  # use half the height of the tallest line
    text_height = text_height_second + text_height_first + vert_space

    # calc a random offset for translation
    rand_height_offset = random_offset(height_offset_range) + height_offset
    rand_width_offset = random_offset(width_offset_range)

    # calc mid position of the text for both lines together
    width_mid_pos = (width-text_width)/2 + rand_width_offset
    height_mid_pos = (height-text_height)/2 + rand_height_offset + vert_space

    # text_color = random_font_color(background_color)  # select a text color based off of background color
    text_color = (0, 0, 0)

    # draw first line
    draw_object.text((width_mid_pos, height_mid_pos + rand_height_offset), first_line, align='center', font=font_obj, fill=text_color)

    # draw second line (need to add spacing this time)
    draw_object.text((width_mid_pos, height_mid_pos + rand_height_offset + vert_space), second_line, align='center', font=font_obj, fill=text_color)


def save_img(img, font_path, i):

    # save image
    img_name = '{}.png'.format(str(i).zfill(3))  # add padding to name and same path name
    img_path = os.path.join(font_path, img_name)
    img = Image.fromarray(img)  # convert back to PIL
    img = img.resize((height, width))
    img.save(img_path)  # save as PIL

    # # optional save a gray img too normalized too, for testing
    # gray_img_name = '{}_gray.png'.format(str(i).zfill(3))
    # gray_img_path = os.path.join(char_root, gray_img_name)
    # save_image(img_path, gray_img_path)


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
        font_root = './{}/{}'.format(root, font_name)  # address_gram_images / {font_name}
        if not os.path.exists(font_root):
            os.mkdir(font_root)

        font_path = fonts_root_dir + '/' + font_files_list[font_idx]   #  path to receive the font file

        # loop thru num of grams for a font
        for n in range(num_grams):

            # background_color = random_back_color()
            background_color = (255, 255, 255)

            # create an image as rgb
            img = Image.new("RGB", (width, height), background_color)

            # select font file and create font object, and set size
            font_obj = ImageFont.truetype(font_path, font_size + random_offset(font_size_range))

            # creates a draw object for im, to allow you to write on the image
            draw_object = ImageDraw.Draw(img)

            # writes the lines onto the image
            write_lines(img, draw_object, font_obj, background_color)

            # draw a random shape on image
            # draw_random_shape(draw_object, background_color)

            img_np = np.asarray(img)  # need to convert image to numpy array

            # aug_img = seq.augment_image(img_np)  # apply augmentation to image

            save_img(img_np, font_root, n)


if __name__ == "__main__":
    # execute only if run as a script
    generate()
