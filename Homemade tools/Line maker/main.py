import os
import cv2
import pandas as pd
import numpy as np
import PySimpleGUI as sg
from PIL import Image

### Configs
sizex, sizey = 640, 480
icon = 'icon.ico' ###Icon made by Freepik -> https://www.flaticon.com/free-icon/picture_937456?term=image%20editing&page=1&position=48
font = 'Arial 12'
back = 'white'
text_color = 'black'

drawing = False
ix, iy = -1, -1

class BBM(object):
    def __init__(self):
        self.img_names = []
        self.target_path = ''
        self.cont = 0
        self.cont2 = 0
        self.cropped_names = []
        self.first = True
        self.p1x, self.p1y, self.p2x, self.p2y = 0, 0, 0, 0
        self.dict = {'P1x': [], 'P1y': [], 'P2x': [], 'P2y': []}
        self.window = self.create_interface()
        self.loop()
        
    def create_interface(self):
        layout = self.create_layout()
        return sg.Window('Bouding Box Maker', layout=layout, size=(sizex, sizey), icon=icon,
                         background_color=back, return_keyboard_events=True, font=font,
                         finalize=True)

    def create_layout(self):
        layout = [ [sg.Text('Select a folder with images', background_color=back, text_color=text_color, font=font)],
                   [sg.Input(key='folder_images'), sg.FolderBrowse(target='folder_images')],
                   [sg.Text('Select a folder to save your work', background_color=back, text_color=text_color, font=font)],
                   [sg.Input(key='folder_save'), sg.FolderBrowse(target='folder_save')],
                   [sg.Text('First image', background_color=back, text_color=text_color, font=font)],
                   [sg.InputCombo(self.img_names, key='first_image', size=(15, 1))],
                   [sg.Button('START', key='start', pad=(150, 100)), sg.Button('STOP', key='stop')] ]

        return layout

    def draw(self, event, x, y, flags, param):
        global ix, iy, drawing
        if (event == cv2.EVENT_LBUTTONDOWN) and not drawing:
            drawing = True
            ix, iy = x, y
            self.p1x = x
            self.p1y = y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                self.img = cv2.imread(self.img_n)
                cv2.line(self.img, (ix, iy), (x, y), (0, 255, 0), 1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            self.img = cv2.imread(self.img_n)
            cv2.line(self.img, (ix, iy), (x, y), (0, 255, 0), 1)
            self.p2x = x
            self.p2y = y

    def edit(self, img_path):
        delete = False
        cv2.namedWindow('Editor')
        cv2.setMouseCallback('Editor', self.draw)
        self.img_n = img_path[self.cont]
        self.img = cv2.imread(self.img_n)

        while True:
            if self.first:
                self.transform(self.img_n)
                self.first = False
                self.cont2 = 0

            cv2.imshow('Editor', self.img)
            k = cv2.waitKey(1) & 0xFF
            ### Enter saves the image with the line and loads next img
            if k == 13:
                self.cont += 1
                self.first = True
                cv2.imwrite(self.values['folder_save'] + '/' + self.img_n[self.img_n.rfind('/') + 1:], self.img)
                self.dict['P1x'].append(self.p1x)
                self.dict['P1y'].append(self.p1y)
                self.dict['P2x'].append(self.p2x)
                self.dict['P2y'].append(self.p2y)

                df = pd.DataFrame(self.dict, columns=list(self.dict.keys()), index=[i[i.rfind('/') + 1:] for i in img_path[:self.cont]])
                df.to_csv(self.values['folder_save'] + '/' + 'values.csv')
                break
            ### Esc stops editing
            if k == 27:
                cv2.destroyAllWindows()
                df = pd.DataFrame(self.dict, columns=list(self.dict.keys()), index=[i[i.rfind('/') + 1:] for i in img_path[:self.cont]])
                df.to_csv(self.values['folder_save'] + '/' + 'values.csv')
                print(self.dict)
                break
            ### Space bar deletes the image and loads next image
            if k == 32:
                delete = True
                self.first = True
                break
        if self.first:
            cv2.destroyAllWindows()
            if delete:
                os.remove(self.img_n)
                img_path.remove(self.img_n)
            self.edit(img_path)

    def transform(self, img_n):
        self.first = False
        img = Image.open(img_n)
        img = img.resize((128, 128), Image.LANCZOS)
        img.save(img_n)

    def loop(self):
        while True:
            event, self.values = self.window.read(timeout=50)

	    ### Close window
            if event in (None, 'Quit', 'Exit'):
                break

            if self.values['folder_images'] != self.target_path:
                self.target_path = self.values['folder_images']
                self.img_names = os.listdir(self.target_path)
                self.window.Element('first_image').update(values=self.img_names)

            if event == 'start':
                img = self.img_names[self.img_names.index(self.values['first_image']) : ]
                img_path = []
                for img_name in img:
                    img_path.append(self.target_path + '/' + img_name)

                if 'cropped' not in os.listdir(self.values['folder_save']):
                    os.mkdir(self.values['folder_save'] + '/' + 'cropped')
                self.cont = 0
                print('Última:', img_path[-1])
                self.edit(img_path)

            if event == 'stop':
                break

if __name__ == '__main__':
    bbm = BBM()

    bbm.window.close()
    cv2.destroyAllWindows()
    #del bbm
