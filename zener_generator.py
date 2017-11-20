from PIL import Image, ImageDraw
import shutil
import random
import os
import sys
import numpy as np


class DataUtil:

    def __init__(self):
        self.color = 0  # color for drawing the image. All images ddrawn in balck
        self.brush_stroke_max = 5  # Max brush stroke
        self.pos_shift_max = 2  # Max shift in position of image shift in either direction as padding is of 2
        self.rotation_max = 90  # Max positive/ negative rotation in degrees orientation transformation
        self.resize_ratio_min = 0.3  # (1- resize_ratio_min) gives lower bound on resized images size
        self.ellip_num_max = 5  # Max no. of stray sllipsoids to be drawn in image
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
        # Maintaining aspect ratio by multiplying height and widht by the same factor
        in_image = in_image.resize((int(width * resize_ratio), int(height * resize_ratio)))
        # Calculate the position where the image is to be pasted so that it is in the center
        diff = (self.size - in_image.size[0])/2
        final_image.paste(in_image, (diff, diff))
        return final_image

    def _trans_pos(self, in_image):
        width, height = in_image.size
        final_image = Image.new('L', (width, height), 255)
        pos_shift = random.randint(-self.pos_shift_max, self.pos_shift_max)
        final_image.paste(in_image, (pos_shift, pos_shift))
        return final_image

    def _trans_orientation(self, in_image):
        rotation_degree = random.randint(-self.rotation_max, self.rotation_max)
        # converted to have an alpha layer
        im2 = in_image.convert('RGBA')
        rot = im2.rotate(rotation_degree)
        # a white image same size as rotated image
        fff = Image.new('RGBA', rot.size, (255,) * 4)
        # create a composite image using the alpha layer of rot as a mask
        final_image = Image.composite(rot, fff, rot)
        final_image = final_image.convert(in_image.mode)
        return final_image

    def _trans_stray(self, in_image):
        ellip_num = random.randint(1, self.ellip_num_max)
        width, height = in_image.size
        draw = ImageDraw.Draw(in_image)
        for i in range(ellip_num):
            ellip_size_x = random.randint(1, self.ellip_size_max)
            ellip_size_y = random.randint(1, self.ellip_size_max)
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            draw.ellipse((x, y, x + ellip_size_x, y + ellip_size_y), fill=self.color, outline=self.color)
        return in_image

    def gen_images(self, folder_name, num_examples):
        symbols = ['o', 'p', 'w', 'q', 's']
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name, True)  # Delete the directory if it is already there
        os.mkdir(folder_name)  # Create directory with the folder name
        os.chdir(folder_name)  # Change the directory so training images are store there
        for i in xrange(num_examples):
            symbol_sel = random.choice(symbols)  # Uniformly randomly select an image form the list
            att = getattr(self, "base_im_" + symbol_sel)  # Get the function object to draw appropriate symbol
            image = att
            image = self._trans_pos(image)
            image = self._trans_resize(image)
            image = self._trans_orientation(image)
            image = self._trans_stray(image)
            image.save(str(i + 1)+'_'+symbol_sel.upper()+".png")

    def gen_images_lett(self, folder_name, num_examples, symbol):
        symbols = ['o', 'p', 'w', 'q', 's']
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name, True)  # Delete the directory if it is already there
        os.mkdir(folder_name)  # Create directory with the folder name
        os.chdir(folder_name)  # Change the directory so training images are store there
        prob = 0.6
        p = [prob if s ==  symbol else (1.0 - prob)/(len(symbols)-1) for s in symbols]
        for i in xrange(num_examples):
            symbol_sel = np.random.choice(a=symbols, size=1, p=p)[-1]  # Generate symbols as per specified probability
            att = getattr(self, "base_im_" + str(symbol_sel))  # Get the function object to draw appropriate symbol
            image = att
            image = self._trans_pos(image)
            image = self._trans_resize(image)
            image = self._trans_orientation(image)
            image = self._trans_stray(image)
            image.save(str(i + 1)+'_'+symbol_sel.upper()+".png")

    def get_data(self, folder_name, symbol_name=""):
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
            x_list.append(image_array)
        x = np.array(x_list, dtype="float64")
        y = np.array(y_list, dtype="float64")
        data = zip(x,y)
        print("No. of P's" + str(np.sum(y)))
        os.chdir('..')
        return Data(data)

class Data:
    def __init__(self, data):
        self.total_data = data
        self.batched_data = list()

    def get_epoch_data(self, batchsize):
        self.batched_data = list()
        if not self.total_data:
            return None
        current_batch = 0
        tot_items = len(self.total_data)
        random.shuffle(self.total_data)
        num_batches = tot_items//batchsize
        while current_batch < num_batches:
            batched_data_x = list()
            batched_data_y = list()
            index_start = current_batch * batchsize
            for i in range(batchsize):
                x, y = self.total_data[index_start + i]
                batched_data_x.append(x)
                batched_data_y.append(y)
            current_batch += 1
            self.batched_data.append((batched_data_x, batched_data_y))
        return self.batched_data

    def get_test_data(self):
        test_x = list()
        test_y = list()
        for i in range(len(self.total_data)):
            x,y = self.total_data[i]
            test_x.append(x)
            test_y.append(y)
        return (test_x,test_y)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Insufficient number of arguments.\nPattern : python zener_generator.py data 10000 symbol")
        sys.exit()
    else:
        fol_name, num_examples, symbol = sys.argv[1:4]
        data = DataUtil()
        print "Data generation in progres.."
        data.gen_images_lett(fol_name, int(num_examples), symbol.lower())
        print "Data generation complete. Images generated to "+ fol_name