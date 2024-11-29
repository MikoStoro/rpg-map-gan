from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QWidget, QFileDialog, QStyle, QColorDialog, QApplication, QFrame
from PyQt6.QtCore import Qt, QStandardPaths
from PyQt6.QtGui import QMouseEvent, QPaintEvent, QPen, QAction, QPainter, QColor, QIcon, QKeySequence, QPixmap

import sys

COLORS = ["#804040", "#008000", "#00B000", "#C0C0C0", "#808080", "#8000FF", "#FFFF80", "#801010", "#400505", "#606060", "#FF8040", "#400040", "#0080FF", "#FF2000", "#FFFFFF"]
TERRAINS = ["Dirt", "Grass", "Plant", "Stone", "Metal", "Magic", "Sand", "Wood", "Manmade Wood", "Manmade Stone", "Manmade Sand", "Roof", "Water", "Lava", "None"]

class PainterWidget(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.default_color = Qt.GlobalColor.white
        
        self.setFixedSize(256, 256)
        self.pixmap = QPixmap(self.size())
        self.pixmap.fill(self.default_color)

        self.colorFrames = []

        self.previous_pos = None
        self.painter = QPainter()
        self.pen = QPen()
        self.pen.setWidth(12)
        self.pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

    def paintEvent(self, e: QPaintEvent) -> None:
        with QPainter(self) as painter:
            painter.drawPixmap(0, 0, self.pixmap)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        self.previous_pos = e.position().toPoint()
        QWidget.mousePressEvent(self, e)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        curr_pos = e.position().toPoint()
        self.painter.begin(self.pixmap)
        self.painter.setRenderHints(QPainter.RenderHint.Antialiasing, True)
        self.painter.setPen(self.pen)
        self.painter.drawLine(self.previous_pos, curr_pos)
        self.painter.end()

        self.previous_pos = curr_pos
        self.update()

        QWidget.mouseMoveEvent(self, e)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self.previous_pos = None
        QWidget.mouseReleaseEvent(self, e)

    def save(self, filename: str):
        self.pixmap.save(filename)

    def load(self, filename: str):
        self.pixmap.load(filename)
        self.pixmap = self.pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.update()

    def clear(self):
        self.pixmap.fill(self.default_color)
        self.update()

    def add_palette_buttons(self, layout: QtWidgets.QVBoxLayout):
        for (c, t) in zip(COLORS, TERRAINS):
            f = QtWidgets.QFrame()
            l = QtWidgets.QHBoxLayout()
            f.setObjectName(t)
            label = QtWidgets.QLabel(t)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignCenter)
            button = QPaletteButton(c)
            button.pressed.connect(lambda t=t: self.set_pen_color(t))

            l.addWidget(label)
            l.addWidget(button)
            f.setLayout(l)
            layout.addWidget(f)
            self.colorFrames.append(f)

    
    def filter_palette_buttons(self, allowed_terrains):
        for f in self.colorFrames:
            f.show()
            if f.objectName() not in allowed_terrains:
                f.hide()

    def set_default_terrain(self, terrain):
        d = {TERRAINS[i]: COLORS[i] for i in range(len(COLORS))}
        try:
            self.default_color = QColor(d[terrain])
        except:
            self.default_color = d["None"]
    
    def underline_terrain(self, name):
        for f in self.colorFrames:
            f.setFrameStyle(QFrame.Shape.NoFrame)
            if f.objectName() == name:
                f.setFrameStyle(QFrame.Shape.Box)

    def set_pen_color(self, terrain):
        d = {TERRAINS[i]: COLORS[i] for i in range(len(COLORS))}
        try:
            self.pen.setColor(QColor(d[terrain]))
            self.underline_terrain(terrain)
        except:
            self.default_color = d["None"]
            self.underline_terrain("None")


    def set_model(self, model):
        pass


class QPaletteButton(QtWidgets.QPushButton):
    def __init__(self, color):
        super().__init__()
        self.setFixedSize(QtCore.QSize(20, 20))
        self.color = color
        self.setStyleSheet("background-color: %s;" % color)

