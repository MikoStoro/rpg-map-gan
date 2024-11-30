import sys

# from box_color_picker import create_mouse_event, create_json_with_colors_and_items

from PyQt6.QtCore import QSize, Qt, QPoint, QFile, QRect
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QApplication, QWidget, QTextEdit, QPushButton, QGridLayout, QLabel, QFileDialog, \
    QHBoxLayout, QLineEdit, QFrame, QVBoxLayout, QComboBox, QSlider, QRadioButton, QButtonGroup
from PainterWidget import PainterWidget

class ModelData:
    def __init__(self, name, path, colors, default_t, default_p):
        self.name = name
        self.path = path
        self.colors = colors
        self.default_terrain = default_t
        self.default_pen = default_p

models = [
     ModelData("Overworld", "./tmp_path", [ "Grass", "Stone", "Sand", "Water", "Dirt"], "Water", "Grass"), 
     ModelData("Caves", "./tmp_path2", ["None", "Stone", "Manmade Stone", "Lava"], "None", "Stone")]

class MainWindow(QWidget):
   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.current_model = models[0]

        self.setWindowTitle("RPG Map Generator (Early Access)")
        #self.setGeometry(100, 100, 1240, 660)

        layout = QGridLayout()
        layout.setSpacing(10)

        # PAINT
        paint_layout = QHBoxLayout()
        self.paint = PainterWidget(self)

        self.palette_layout = QVBoxLayout()
        self.paint.add_palette_buttons(self.palette_layout)

        paint_layout.addLayout(self.palette_layout)
        paint_layout.addWidget(self.paint)
        layout.addLayout(paint_layout, 0, 0)


        self.output_image = QLabel()
        self.output_image.setFixedSize(256, 256)
        self.output_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_image.setFrameStyle(QFrame.Shape.Box)
        layout.addWidget(self.output_image, 0, 1)
        
        self.toolbar = QHBoxLayout()
        layout.addLayout(self.toolbar, 1, 0, 1,2)
        
        self.model_select = QComboBox()
        self.model_select.currentIndexChanged.connect(lambda: self.switch_model())
        self.populate_model_select()
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(lambda: self.paint.clear())
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(lambda: self.process())

        self.brushSize = QSlider()
        self.brushSize.setOrientation(Qt.Orientation.Horizontal)
        self.brushSize.setValue(15)
        self.brushSize.setMaximum(150)
        self.brushSize.setMinimum(1)
        self.brushSize.valueChanged.connect(self.paint.set_pen_size)

        self.brush_type = QRadioButton("circle")
        self.brush_type.setChecked(True)
        self.brush_type.toggled.connect(self.paint.set_pen_shape)
        self.brush_type2 = QRadioButton("square")
        self.brush_type2.toggled.connect(self.paint.set_pen_shape)
        self.bruh_group = QButtonGroup()
        self.bruh_group.addButton(self.brush_type)
        self.bruh_group.addButton(self.brush_type2)

        self.canvas_size = QRadioButton("256")
        self.canvas_size.setChecked(True)
        self.canvas_size.toggled.connect(self.set_size)
        self.canvas_size2 = QRadioButton("512")
        self.canvas_size2.toggled.connect(self.set_size)
        self.canvas_group = QButtonGroup()
        self.canvas_group.addButton(self.canvas_size)
        self.canvas_group.addButton(self.canvas_size2)

        self.toolbar.addWidget(self.canvas_size)
        self.toolbar.addWidget(self.canvas_size2)
        self.toolbar.addWidget(self.brush_type)
        self.toolbar.addWidget(self.brush_type2)
        self.toolbar.addWidget(self.brushSize)
        self.toolbar.addWidget(self.model_select)
        self.toolbar.addWidget(self.reset_button)
        self.toolbar.addWidget(self.process_button)



        self.setLayout(layout)
        self.show()
        self.refresh_palette()
        self.paint.set_default_terrain(self.current_model.default_terrain)
        self.setMouseTracking(True)

    def populate_model_select(self):
        for m in models:
            self.model_select.addItem(m.name)

    def load_model(self, path):
        pass

    def switch_model(self):
        self.current_model = models[self.model_select.currentIndex()]
        self.refresh_palette()
        self.paint.set_default_terrain(self.current_model.default_terrain)
        self.paint.clear()
        self.paint.set_pen_color(self.current_model.default_pen)
        pass 

    def process(self):
        size = self.paint.size().width()
        cuts = size // 256
        for i in range(cuts):
            for j in range(cuts):
                rect = QRect(i*256, j*256, 256, 256)
                cut = self.paint.pixmap.copy(rect)
                cut.save("tmp"+str(i)+str(j)+".png")

    def refresh_palette(self):
        self.paint.filter_palette_buttons(self.current_model.colors)

    def set_size(self):
        value = int(self.sender().text())
        self.output_image.setFixedSize(value, value)
        self.paint.setFixedSize(value,value)
        self.paint.new_size()
        self.adjustSize()

if __name__ == "__main__":
    print("Starting...")

    app = QApplication(sys.argv)
    print("Application started.")
    window = MainWindow()
    print("Window created.")
    sys.exit(app.exec())

    print("Done.")