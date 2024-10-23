import sys

import json
import PIL
from PIL import Image, ImageDraw, ImageFilter

# from box_color_picker import create_mouse_event, create_json_with_colors_and_items
from scanner import get_map_with_scan_overlay, scan_map, save_map_with_scan_overlay

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QApplication, QWidget, QTextEdit, QPushButton, QGridLayout, QLabel, QFileDialog, QHBoxLayout
import cv2

DEFAULT_COLORS = '{\n    "S": [128, 128, 128],\n    "K": [0, 0, 0],\n    "W": [50, 50, 255],\n    "D": [139, 69, 19],\n    "G": [50, 140, 50]\n}'
TMP_IMG_PATH = "./tmp.png"
DEFAULT_DIR = "./Original Sin II"

class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_image = ""

        self.setWindowTitle("RPG Map Generator (Early Access)")
        self.setGeometry(100, 100, 1240, 660)

        layout = QGridLayout()
        layout.setSpacing(10)

        self.json = QTextEdit()
        self.json.setPlainText(DEFAULT_COLORS)
        self.json.setFixedSize(200, 600)
        layout.addWidget(self.json, 0, 0)

        self.input_image = QLabel()
        self.input_image.setFixedSize(500, 600)
        layout.addWidget(self.input_image, 0, 1)

        self.output_image = QLabel()
        self.output_image.setFixedSize(500, 600)
        layout.addWidget(self.output_image, 0, 2)

        left_button_panel = QHBoxLayout()
        self.save_json = QPushButton("Zapisz")
        self.save_json.setFixedSize(80, 30)
        self.save_json.clicked.connect(lambda: self.save_json_dialog())
        left_button_panel.addWidget(self.save_json)

        self.open_json = QPushButton("Otwórz")
        self.open_json.setFixedSize(80, 30)
        self.open_json.clicked.connect(lambda: self.open_json_dialog())
        left_button_panel.addWidget(self.open_json)
        layout.addLayout(left_button_panel, 1, 0)

        middle_button_panel = QHBoxLayout()
        self.open_input = QPushButton("Otwórz")
        self.open_input.setFixedSize(100, 30)
        self.open_input.clicked.connect(lambda: self.open_image_dialog())
        middle_button_panel.addWidget(self.open_input)

        self.classify = QPushButton("Klasyfikuj")
        self.classify.setFixedSize(100, 30)
        self.classify.clicked.connect(lambda : self.run_classify())
        middle_button_panel.addWidget(self.classify)
        layout.addLayout(middle_button_panel, 1, 1)

        self.save_output = QPushButton("Fidget button")
        self.save_output.setFixedSize(100, 30)
        self.save_output.clicked.connect(lambda: self.save_image_dialog())
        layout.addWidget(self.save_output, 1, 2)

        self.setLayout(layout)
        self.show()

    def save_json_dialog(self):
        filename, ok = QFileDialog.getSaveFileName(self, "Zapisz plik", "../", "Pliki .json (*json)")
        if ok:
            with open(filename, 'w') as file:
                file.write(str(self.json.toPlainText()))
            print(f"Pallette saved to {filename}.")
        else:
            print("Pallette save cancelled.")

    def open_json_dialog(self):
        filename, ok = QFileDialog.getOpenFileName(self, "Wybierz plik", "../", "Pliki .json (*json)")
        if ok:
            with open(filename, 'r') as file:
                text = file.read()
                print(text)
                self.json.setPlainText(text)
            print("Pallette opened.")
        else:
            print("Pallette open cancelled.")


    def open_image_dialog(self):
        filename, ok = QFileDialog.getOpenFileName(self, "Wybierz plik", DEFAULT_DIR, "Obrazy (*.png *.jpg)")
        if ok:
            self.input_image.setPixmap(QPixmap(filename))
            self.current_image = filename
            print("Image opened.")
        else:
            print("Image open cancelled.")

    def run_classify(self):
        print("Classification started.")
        defined_colours = json.loads(str(self.json.toPlainText()))
        for name in defined_colours.keys():
            defined_colours[name] = tuple(defined_colours[name])
        defined_colours = dict((value, key) for key, value in defined_colours.items())
        print(defined_colours)

        color_matrix = scan_map(self.current_image, defined_colours) 
        result = Image.fromarray(get_map_with_scan_overlay(self.current_image, color_matrix, defined_colours))

        '''filename, ok = QFileDialog.getSaveFileName(self, "Zapisz plik", "../", "Obrazy (*.png *.jpg)")
        if ok:
            result.save(filename)
            self.output_image.setPixmap(QPixmap(filename))
            print("Result saved.")
        else:
            print("Result save cancelled.")'''
        result.save(TMP_IMG_PATH)
        self.output_image.setPixmap(QPixmap(TMP_IMG_PATH))


    def save_image_dialog(self):
        print("POGGERS")


if __name__ == "__main__":
    print("Starting...")

    app = QApplication(sys.argv)
    print("Application started.")
    window = MainWindow()
    print("Window created.")
    sys.exit(app.exec())

    print("Done.")
