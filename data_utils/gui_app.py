import sys
import utils
import json
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os

# from box_color_picker import create_mouse_event, create_json_with_colors_and_items
from scanner import get_map_with_scan_overlay, scan_map, save_map_with_scan_overlay, dual_sliding_window_matrices_no_padding

from PyQt6.QtCore import QSize, Qt, QPoint
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QApplication, QWidget, QTextEdit, QPushButton, QGridLayout, QLabel, QFileDialog, QHBoxLayout, QLineEdit, QFrame,QVBoxLayout
import cv2
import box_color_picker as picker
import colormap_createor
from pathlib import Path
from PainterWidget import PainterWidget

DEFAULT_COLORS = '{\n    "S": [128, 128, 128],\n    "K": [0, 0, 0],\n    "W": [50, 50, 255],\n    "D": [139, 69, 19],\n    "G": [50, 140, 50]\n}'
TMP_IMG_PATH = "./tmp.png"
TMP_SLICE_PATH = "./tmp_slice.png"
DEFAULT_DIR = "./Original Sin II"
DEFAULT_JSON_PATH = "./tmp.json"
#RESULTS_DIR = "./results"
RESULTS_DIR = "./debug_results"

import matplotlib.pyplot as plt


class MainWindow(QWidget):
   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_image_path = ""

        self.GRID_SIZE = None

        self.slice_point_1 = None
        self.slice_point_2 = None

        self.current_slice = None

        self.current_image = None
        self.current_result = None

        self.json_path = DEFAULT_JSON_PATH

        self.label_matrix = None
        self.last_grid_size = 2

        self.setWindowTitle("RPG Map Generator (Early Access)")
        self.setGeometry(100, 100, 1240, 660)

        layout = QGridLayout()
        layout.setSpacing(10)

        self.json = QTextEdit()
        self.json.textChanged.connect(lambda : self.save_json_file(self.json_path))
        self.load_tmp_json()
        self.json.setFixedSize(200, 600)
        layout.addWidget(self.json, 0, 0)

        
        slice_manager_panel = QVBoxLayout()
        
        self.current_slice_label = QLabel("Slice goes here")
        slice_manager_panel.addWidget(self.current_slice_label)

        self.slice_name = QLineEdit()
        self.slice_name.setPlaceholderText("Slice Name")
        slice_manager_panel.addWidget(self.slice_name)

        self.slice_accept = QPushButton("Zatwierdź")
        self.slice_accept.clicked.connect(lambda : self.add_json_element())
        slice_manager_panel.addWidget(self.slice_accept)

        layout.addLayout(slice_manager_panel,0,1)

        self.input_image = QLabel()
        self.input_image.setFixedSize(500, 600)
        self.input_image.setPixmap(QPixmap())
        self.input_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ##self.input_image.setMouseTracking(True)
        self.input_image.setFrameStyle(QFrame.Shape.Box)
        # layout.addWidget(self.input_image, 0, 2)

        # PAINT
        paint_layout = QHBoxLayout()
        self.paint = PainterWidget(self)

        palette = QVBoxLayout()
        self.paint.add_palette_buttons(palette)

        paint_layout.addLayout(palette)
        paint_layout.addWidget(self.paint)
        layout.addLayout(paint_layout, 0, 2)


        self.output_image = QLabel()
        self.output_image.setFixedSize(500, 600)
        self.output_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_image.setFrameStyle(QFrame.Shape.Box)
        layout.addWidget(self.output_image, 0, 3)

        left_button_panel = QHBoxLayout()
        self.save_json = QPushButton("Zapisz JSON")
        self.save_json.setFixedSize(80, 30)
        self.save_json.clicked.connect(lambda: self.save_json_dialog())
        left_button_panel.addWidget(self.save_json)

        self.open_json = QPushButton("Otwórz JSON")
        self.open_json.setFixedSize(80, 30)
        self.open_json.clicked.connect(lambda: self.open_json_dialog())
        left_button_panel.addWidget(self.open_json)
        layout.addLayout(left_button_panel, 1, 0)

        middle_button_panel = QHBoxLayout()
        self.open_input = QPushButton("Otwórz Obraz")
        self.open_input.setFixedSize(100, 30)
        self.open_input.clicked.connect(lambda: self.open_image_dialog())
        middle_button_panel.addWidget(self.open_input)

        self.classify = QPushButton("Klasyfikuj")
        self.classify.setFixedSize(100, 30)
        self.classify.clicked.connect(lambda : self.run_classify())
        middle_button_panel.addWidget(self.classify)
        layout.addLayout(middle_button_panel, 1, 2)

        self.grid_size_input = QLineEdit()
        self.grid_size_input.setText("2")
        self.grid_size_input.setPlaceholderText("Grid Size")
        self.grid_size_input.setFixedSize(100, 30)
        self.grid_size_input.textChanged.connect(lambda : self.set_grid_size())
        self.set_grid_size()
        middle_button_panel.addWidget(self.grid_size_input)
        layout.addLayout(middle_button_panel, 1, 2)


        self.save_output = QPushButton("Zapisz wynik")
        self.save_output.setFixedSize(100, 30)
        self.save_output.clicked.connect(lambda: self.save_image_dialog())
        layout.addWidget(self.save_output, 1, 3)

        self.clicked_terrain = QLabel("Clicked terrain")
        layout.addWidget(self.clicked_terrain, 1, 4)
        
        self.setLayout(layout)
        self.show()

        self.setMouseTracking(True)

    def load_tmp_json(self):
        try:
            self.open_json_file(DEFAULT_JSON_PATH)
        except:
            open(DEFAULT_JSON_PATH, "w").close()
            self.open_json_file(DEFAULT_JSON_PATH)

    def add_json_element(self):
        name = self.slice_name.text()
        slice = self.current_slice
        picker.add_item_to_json(name,slice,self.json_path)
        self.open_json_file(self.json_path)
    


    def get_label(self, x,y):
        label_x = self.output_image.x() 
        label_y = self.output_image.y()
        label_w = self.output_image.width()
        label_h = self.output_image.height()


        pixmap_w = self.output_image.pixmap().width()
        pixmap_h = self.output_image.pixmap().height()
        pixmap_x = label_w/2 - pixmap_w/2
        pixmap_y = label_h/2 - pixmap_h/2
        
        x_relative_to_label = x - label_x
        y_relative_to_label = y - label_y

        x_relative_to_pixmap = int(x_relative_to_label - pixmap_x)
        y_relative_to_pixmap = int(y_relative_to_label - pixmap_y)

        if(x_relative_to_label < 0 or 
           x_relative_to_label > self.output_image.width() or
           y_relative_to_label < 0 or
           y_relative_to_label > self.output_image.height()):
            return
        if(y_relative_to_pixmap>= 0 and y_relative_to_pixmap < self.label_matrix.shape[0]
           and x_relative_to_pixmap>= 0 and x_relative_to_pixmap < self.label_matrix.shape[1]):
            print("label: "  + str(x_relative_to_pixmap),str(y_relative_to_pixmap))
            label = self.label_matrix[y_relative_to_pixmap][x_relative_to_pixmap]
            print(label)
            self.clicked_terrain.setText(label)

    def mousePressEvent(self,e):

        x = int(e.position().x())
        y = int(e.position().y())
        self.get_label(x,y)

        label_x = self.input_image.x() 
        label_y = self.input_image.y()
        label_w = self.input_image.width()
        label_h = self.input_image.height()


        pixmap_w = self.input_image.pixmap().width()
        pixmap_h = self.input_image.pixmap().height()
        pixmap_x = label_w/2 - pixmap_w/2
        pixmap_y = label_h/2 - pixmap_h/2
        
        x_relative_to_label = x - label_x
        y_relative_to_label = y - label_y

        x_relative_to_pixmap = int(x_relative_to_label - pixmap_x)
        y_relative_to_pixmap = int(y_relative_to_label - pixmap_y)

        if(x_relative_to_label < 0 or 
           x_relative_to_label > self.input_image.width() or
           y_relative_to_label < 0 or
           y_relative_to_label > self.input_image.height()):
            return
        
        if self.slice_point_1 is None:
            self.slice_point_1 = QPoint(x_relative_to_pixmap, y_relative_to_pixmap)
        else:
            self.slice_point_2 = QPoint(x_relative_to_pixmap, y_relative_to_pixmap)
            try:
                returned_slice = picker.get_slice(self.current_image_path, 
                             self.slice_point_1.x(), self.slice_point_1.y(), 
                             self.slice_point_2.x(), self.slice_point_2.y())
                self.current_slice = returned_slice
                #returned_slice.save(TMP_SLICE_PATH)
                #self.current_slice.setPixmap(QPixmap(TMP_SLICE_PATH))
                #print("debug")
                
                try:
                    returned_slice.save(TMP_SLICE_PATH)
                    self.current_slice_label.setPixmap(QPixmap(TMP_SLICE_PATH))
                except Exception as e:
                    print(e)

            except:
                print("Invalid coordinates chosen")
            self.slice_point_1 = None
            self.slice_point_2 = None
       
        print(str(x) + " " + str(y))
        print(str(x_relative_to_label) + " " + str(y_relative_to_label))
        print(str(x_relative_to_pixmap) + " " + str(y_relative_to_pixmap))

    def set_grid_size(self):
        raw_input = self.grid_size_input.text().strip()
        if raw_input is None or raw_input == "":
            self.GRID_SIZE = None
        try:
            grid_size = int(raw_input)
            self.GRID_SIZE = grid_size
            print("Grid size changed: " + str(self.GRID_SIZE))
        except:
            self.grid_size_input.clear()
            self.GRID_SIZE = None

    def save_json_dialog(self):
        filename, ok = QFileDialog.getSaveFileName(self, "Zapisz plik", "../", "Pliki .json (*json)")
        if ok:
            with open(filename, 'w') as file:
                file.write(str(self.json.toPlainText()))
            print(f"Pallette saved to {filename}.")
        else:
            print("Pallette save cancelled.")

    def open_json_file(self, filename):
        try:
            with open(filename, 'r') as file:
                text = file.read()
                text = text.replace(',', ',\n')
                text = text.replace('\n\n', '\n')
                self.json.setPlainText(text)

        except:
           open(filename, 'w').close()
           self.open_json_file(filename)

    def save_json_file(self, filename):
        with open(filename, "w") as f: 
            f.write(self.json.toPlainText())
    
    def open_json_dialog(self):
        filename, ok = QFileDialog.getOpenFileName(self, "Wybierz plik", "../", "Pliki .json (*json)")
        if ok:
            self.json_path = filename
            self.open_json_file(filename)
            print("Pallette opened.")
        else:
            print("Pallette open cancelled.")


    def open_image_dialog(self):
        filename, ok = QFileDialog.getOpenFileName(self, "Wybierz plik", DEFAULT_DIR, "Obrazy (*.png *.jpg)")
        if ok:
            # self.input_image.setPixmap(QPixmap(filename))
            self.paint.load(filename)
            self.current_image_path = filename
            self.current_image = np.asarray(Image.open(filename))
            self.current_result = None
            print("Image opened.")
        else:
            print("Image open cancelled.")


    def run_classify(self):
        print("Classification started.")
        defined_colours = json.loads(str(self.json.toPlainText()))
        defined_colours = utils.translate_json_to_colors_dict(defined_colours)
        print("defined colors" + str(defined_colours))

        label_matrix = scan_map(self.current_image_path, defined_colours, grid_size=self.GRID_SIZE) 
        self.label_matrix = label_matrix
        self.last_grid_size = self.GRID_SIZE
        print("label matrix size: " + str(label_matrix.shape))
        print(label_matrix)
       
        #print(colormap.shape)
       
        
        map_with_overlay = get_map_with_scan_overlay(self.current_image_path, label_matrix, defined_colours, opacity=0.9, grid_size=1)
        print("original image size: " + str(map_with_overlay.shape))
        result = Image.fromarray(map_with_overlay)
        self.current_result = label_matrix
        result.save(TMP_IMG_PATH)
        self.output_image.setPixmap(QPixmap(TMP_IMG_PATH))

    def save_image_dialog(self):
        if self.current_image is None:
            print("CHOOSE IMAGE FIRST")
            return
        if self.current_result is None:
            print("GENERATE RESULT FIRST")
            return
        results_directory = RESULTS_DIR
        os.makedirs(results_directory, exist_ok=True)
        name = Path(self.current_image_path).stem
        image_name = results_directory + "/" + name + "_ORIGINAL"
        colormap_name = results_directory + "/" + name + "_COLORMAP"

        defined_colours = json.loads(str(self.json.toPlainText()))
        defined_colours = utils.translate_json_to_colors_dict(defined_colours)
        label_matrix = scan_map(self.current_image_path, defined_colours, grid_size=1) 
        print("SAVED DIMS")
        print(self.current_image.shape)
        colormap = colormap_createor.get_colormap(label_matrix)
        print(colormap.shape)
        Image.fromarray(colormap).save("./tmp_colormap.png")
        np.save(image_name, self.current_image)
        np.save(colormap_name, colormap)

        orig_slices, colormap_slices = dual_sliding_window_matrices_no_padding(self.current_image, colormap, window_size=256, step_size=64)
        print("SLICES")
        print(orig_slices.shape)
        print(colormap_slices.shape)
        for i in range(len(orig_slices)):
            np.save(image_name + "_" + str(i), orig_slices[i])
        for i in range(len(colormap_slices)):
            np.save(colormap_name + "_" + str(i), colormap_slices[i])
        
            

        


if __name__ == "__main__":
    print("Starting...")

    app = QApplication(sys.argv)
    print("Application started.")
    window = MainWindow()
    print("Window created.")
    sys.exit(app.exec())

    print("Done.")
