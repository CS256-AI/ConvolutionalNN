from PIL import Image, ImageDraw
import shutil
import random
import os
import sys
import numpy as np


class Data:

    def __init__(self):
        self.color = 0  # color for drawing the image. All images ddrawn in balck
        self.brush_stroke_max = 2  # Max brush stroke
        self.pos_shift_max = 2  # Max shift in position of image shift in either direction as padding is of 2
        self.rotation_max = 25  # Max positive/ negative rotation in degrees orientation transformation
        self.resize_ratio_min = 0.1  # (1- resize_ratio_min) gives lower bound on resized images size
        self.ellip_num_max = 3  # Max no. of stray sllipsoids to be drawn in image
        self.ellip_size_max = 2  # Max width, height of ellipsoid in pixels
        self.size = 25  # size of canvas
        self.base_im_o = self._draw_o()
        self.base_im_p = self._draw_p()
        self.base_im_w = self._draw_w()
        self.base_im_s = self._draw_s()
        self.base_im_q = self._draw_q()

    def _draw_o(self):
        image = Image.open('o.png')
        image = image.convert(mode='L')
        final_image = Image.new('L', (self.size, self.size), 255)
        diff = (self.size - image.size[0]) / 2
        final_image.paste(image, (diff, diff))
        # final_image.show()
        return final_image

    def _draw_p(self):
        image = Image.open('p.png')
        image = image.convert(mode='L')
        final_image = Image.new('L', (self.size, self.size), 255)
        diff = (self.size - image.size[0]) / 2
        final_image.paste(image, (diff, diff))
        # final_image.show()
        return final_image

    def _draw_q(self):
        image = Image.open('q.png')
        image = image.convert(mode='L')
        final_image = Image.new('L', (self.size, self.size), 255)
        diff = (self.size - image.size[0]) / 2
        final_image.paste(image, (diff, diff))
        # final_image.show()
        return final_image

    def _draw_s(self):
        image = Image.open('s.png')
        image = image.convert(mode='L')
        final_image = Image.new('L', (self.size, self.size), 255)
        diff = (self.size - image.size[0]) / 2
        final_image.paste(image, (diff, diff))
        # final_image.show()
        return final_image

    def _draw_w(self):
        image = Image.open('w.png')
        image = image.convert(mode='L')
        final_image = Image.new('L', (self.size, self.size), 255)
        diff = (self.size - image.size[0]) / 2
        final_image.paste(image, (diff, diff))
        # final_image.show()
        return final_image

    def _trans_resize(self, in_image):
        width, height = in_image.size
        final_image = Image.new('L', (self.size, self.size), 255)
        resize_ratio = 1 - random.random() * self.resize_ratio_min
        # print('resize factor --> {}'.format(resize_ratio))
        # print((width, height))
        # print((int(width * resize_ratio), int(height * resize_ratio)))
        # Maintaining aspect ratio by multiplying height and widht by the same factor
        in_image = in_image.resize((int(width * resize_ratio), int(height * resize_ratio)))

        # print(numpy.array(in_image))
        # print(in_image)
        # print(in_image.size)
        # in_image.show()
        # Calculate the position where the image is to be pasted so that it is in the center
        diff = (self.size - in_image.size[0])/2
        final_image.paste(in_image, (diff, diff))
        # print(final_image.size)
        # print('trans_resize img mode --> {}'.format(final_image.mode))
        return final_image

    def _trans_pos(self, in_image):
        width, height = in_image.size
        final_image = Image.new('L', (width, height), 255)
        pos_shift = random.randint(-self.pos_shift_max, self.pos_shift_max)
        # print('pos_shift --> {}'.format(pos_shift))
        final_image.paste(in_image, (pos_shift, pos_shift))
        # print('_trans_pos img mode --> {}'.format(final_image.mode))
        return final_image

    def _trans_orientation(self, in_image):
        # print('_trans_orientation input img mode --> {}'.format(in_image.mode))
        rotation_degree = random.randint(-self.rotation_max, self.rotation_max)
        # print('rotation_degree --> {}'.format(rotation_degree))
        # converted to have an alpha layer
        im2 = in_image.convert('RGBA')
        # im2.show()
        rot = im2.rotate(rotation_degree)
        # rot.show()
        # a white image same size as rotated image
        fff = Image.new('RGBA', rot.size, (255,) * 4)
        # create a composite image using the alpha layer of rot as a mask
        final_image = Image.composite(rot, fff, rot)
        final_image = final_image.convert(in_image.mode)
        # out_image.show()
        # print('_trans_orientation img mode --> {}'.format(final_image.mode))
        return final_image

    def _trans_stray(self, in_image):
        ellip_num = random.randint(1, self.ellip_num_max)
        # print('ellip_num --> {}'.format(ellip_num))
        width, height = in_image.size
        draw = ImageDraw.Draw(in_image)
        for i in range(ellip_num):
            ellip_size_x = random.randint(1, self.ellip_size_max)
            ellip_size_y = random.randint(1, self.ellip_size_max)
            # print('ellip_size_x --> {}'.format(ellip_size_x))
            # print('ellip_size_y --> {}'.format(ellip_size_y))
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            draw.ellipse((x, y, x + ellip_size_x, y + ellip_size_y), fill=self.color, outline=self.color)
        # print('_trans_stray img mode --> {}'.format(in_image.mode))
        return in_image

    def gen_images(self, folder_name, num_examples):
        symbols = ['o', 'p', 'w', 'q', 's']
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name, True)  # Delete the directory if it is already there
        os.mkdir(folder_name)  # Create directory with the folder name
        os.chdir(folder_name)  # Change the directory so training images are store there
        for i in xrange(num_examples):
            symbol_sel = random.choice(symbols)  # Uniformly randomly select an image form the list
            # print("func --> {}".format(func))
            # The variation for thickness of brush is ensured by random value of width
            # width = random.randint(1, self.brush_stroke_max)
            att = getattr(self, "base_im_" + symbol_sel)  # Get the function object to draw appropriate symbol
            image = att
            #print('Image size --> {}'.format(image.size))
            image = self._trans_pos(image)
            image = self._trans_resize(image)
            image = self._trans_orientation(image)
            image = self._trans_stray(image)
            image.save(str(i + 1)+'_'+symbol_sel.upper()+".png")

    def get_data(self, folder_name, symbol_name=""):
        file_names = []
        y_list = []  # label for image
        x_list = []  # input matrix containing image data
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        os.chdir(folder_name)
        symbol_name = symbol_name.upper()
        for file_name in os.listdir('.'):
            if not (file_name.endswith('.png')):
                continue
            image_array = np.array(Image.open(file_name))
            if symbol_name in file_name:
                y_list.append(1)
            else:
                y_list.append(0)
            file_names.append(file_name)
            x_list.append(image_array)
        x = np.array(x_list, dtype="float64")
        # x.reshape(x.shape[0], x.shape[1], x.shape[2])
        y = np.array(y_list, dtype="float64")
        return (file_names,x,y)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Insufficient number of arguments.\nPattern : python zener_generator.py data 10000")
        sys.exit()
    else:
        fol_name, num_examples = sys.argv[1:3]
        data = Data()
        print "Data generation in progres.."
        data.gen_images(fol_name, int(num_examples))
        print "Data generation complete. Images generated to "+ fol_name