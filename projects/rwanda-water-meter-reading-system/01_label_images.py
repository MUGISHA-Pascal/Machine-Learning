#!/usr/bin/env python3
"""
01_label_images.py

Water Meter YOLO Labeler
"""

import os
import sys
import shutil
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QInputDialog,
    QDialog,
)


CLASS_NAMES = [
    "meter", "window",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "unknown",
]

CLASS_COLORS = {
    0: QColor(0, 255, 0),
    1: QColor(255, 0, 0),
    2: QColor(255, 255, 0),
    3: QColor(255, 165, 0),
    4: QColor(0, 255, 255),
    5: QColor(255, 0, 255),
    6: QColor(128, 255, 0),
    7: QColor(0, 128, 255),
    8: QColor(255, 128, 128),
    9: QColor(128, 0, 255),
    10: QColor(200, 200, 255),
    11: QColor(255, 200, 200),
    12: QColor(180, 180, 180),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HANDLE_SIZE = 10
MIN_BOX_SIZE = 5


def list_image_files(folder: str) -> List[str]:
    return sorted([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])


def unique_destination_path(destination_path: str) -> str:
    if not os.path.exists(destination_path):
        return destination_path

    folder = os.path.dirname(destination_path)
    filename = os.path.basename(destination_path)
    stem, ext = os.path.splitext(filename)

    counter = 1

    while True:
        new_path = os.path.join(folder, f"{stem}_removed_{counter}{ext}")
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def rotate_image_keep_size_crop_edges(image, angle_degrees: float):
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def crop_image(image, x1, y1, x2, y2):
    h, w = image.shape[:2]

    x_min = int(max(0, min(x1, x2)))
    y_min = int(max(0, min(y1, y2)))
    x_max = int(min(w, max(x1, x2)))
    y_max = int(min(h, max(y1, y2)))

    if x_max <= x_min or y_max <= y_min:
        return None

    return image[y_min:y_max, x_min:x_max]


def safe_message(parent, title: str, message: str, button_text: str = "OK") -> None:
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setModal(True)
    dialog.resize(520, 170)

    label = QLabel(message)
    label.setWordWrap(True)

    ok_btn = QPushButton(button_text)
    ok_btn.clicked.connect(dialog.accept)

    buttons = QHBoxLayout()
    buttons.addStretch()
    buttons.addWidget(ok_btn)

    layout = QVBoxLayout()
    layout.addWidget(label)
    layout.addStretch()
    layout.addLayout(buttons)

    dialog.setLayout(layout)
    dialog.exec()


def safe_confirm(parent, title: str, message: str, yes_text: str = "Yes", no_text: str = "Cancel") -> bool:
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setModal(True)
    dialog.resize(560, 200)

    label = QLabel(message)
    label.setWordWrap(True)

    yes_btn = QPushButton(yes_text)
    no_btn = QPushButton(no_text)

    yes_btn.clicked.connect(dialog.accept)
    no_btn.clicked.connect(dialog.reject)

    buttons = QHBoxLayout()
    buttons.addStretch()
    buttons.addWidget(yes_btn)
    buttons.addWidget(no_btn)

    layout = QVBoxLayout()
    layout.addWidget(label)
    layout.addStretch()
    layout.addLayout(buttons)

    dialog.setLayout(layout)

    return dialog.exec() == QDialog.Accepted


@dataclass
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    def copy(self) -> "Box":
        return Box(self.class_id, self.x1, self.y1, self.x2, self.y2)

    def normalized(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        x_min = min(self.x1, self.x2)
        y_min = min(self.y1, self.y2)
        x_max = max(self.x1, self.x2)
        y_max = max(self.y1, self.y2)

        x_center = ((x_min + x_max) / 2.0) / img_w
        y_center = ((y_min + y_max) / 2.0) / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h

        return x_center, y_center, width, height

    @staticmethod
    def from_normalized(class_id, x_center, y_center, width, height, img_w, img_h) -> "Box":
        bw = width * img_w
        bh = height * img_h
        cx = x_center * img_w
        cy = y_center * img_h

        return Box(
            class_id,
            cx - bw / 2.0,
            cy - bh / 2.0,
            cx + bw / 2.0,
            cy + bh / 2.0,
        )

    def rect(self):
        return (
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
        )

    def contains(self, x, y) -> bool:
        x_min, y_min, x_max, y_max = self.rect()
        return x_min <= x <= x_max and y_min <= y <= y_max

    def width(self):
        return abs(self.x2 - self.x1)

    def height(self):
        return abs(self.y2 - self.y1)

    def is_too_small(self, min_size: int = MIN_BOX_SIZE) -> bool:
        return self.width() < min_size or self.height() < min_size

    def move(self, dx, dy, img_w, img_h) -> None:
        x_min, y_min, x_max, y_max = self.rect()
        box_w = x_max - x_min
        box_h = y_max - y_min

        new_x_min = max(0, min(img_w - box_w, x_min + dx))
        new_y_min = max(0, min(img_h - box_h, y_min + dy))

        self.x1 = new_x_min
        self.y1 = new_y_min
        self.x2 = new_x_min + box_w
        self.y2 = new_y_min + box_h

    def clamp(self, img_w, img_h) -> None:
        self.x1 = max(0, min(img_w - 1, self.x1))
        self.y1 = max(0, min(img_h - 1, self.y1))
        self.x2 = max(0, min(img_w - 1, self.x2))
        self.y2 = max(0, min(img_h - 1, self.y2))


class ImageCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.main_window = None

        self.base_pixmap = None
        self.display_pixmap = None

        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.mode = "idle"
        self.crop_mode_enabled = False
        self.resize_handle = None

        self.start_widget_point = None
        self.current_widget_point = None
        self.last_img_point = None

        self.original_box_before_edit = None

    def set_main_window(self, window):
        self.main_window = window

    def set_image(self, qpixmap: QPixmap):
        self.base_pixmap = qpixmap
        self.update_scaled_pixmap()

    def clear_image(self):
        self.base_pixmap = None
        self.display_pixmap = None
        self.clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        if self.base_pixmap is None:
            self.clear()
            return

        label_w = max(1, self.width())
        label_h = max(1, self.height())

        self.display_pixmap = self.base_pixmap.scaled(
            label_w,
            label_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.offset_x = (label_w - self.display_pixmap.width()) // 2
        self.offset_y = (label_h - self.display_pixmap.height()) // 2
        self.scale = self.display_pixmap.width() / self.base_pixmap.width()

        canvas = QPixmap(label_w, label_h)
        canvas.fill(Qt.black)

        painter = QPainter(canvas)
        painter.drawPixmap(self.offset_x, self.offset_y, self.display_pixmap)

        if self.main_window is not None:
            self.main_window.draw_boxes(painter, self.scale, self.offset_x, self.offset_y)

        if self.mode in ("drawing", "crop") and self.start_widget_point and self.current_widget_point:
            if self.mode == "crop":
                draft_color = QColor(0, 255, 255)
            else:
                draft_color = CLASS_COLORS.get(
                    self.main_window.current_class_id,
                    QColor(255, 255, 255),
                )

            painter.setPen(QPen(draft_color, 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)

            rect = QRectF(self.start_widget_point, self.current_widget_point).normalized()
            painter.drawRect(rect)

        painter.end()
        self.setPixmap(canvas)

    def widget_to_image(self, point: QPointF):
        if self.base_pixmap is None or self.display_pixmap is None:
            return None

        x = point.x() - self.offset_x
        y = point.y() - self.offset_y

        if x < 0 or y < 0 or x > self.display_pixmap.width() or y > self.display_pixmap.height():
            return None

        return x / self.scale, y / self.scale

    def mousePressEvent(self, event):
        if self.base_pixmap is None or self.main_window is None:
            return

        if event.button() != Qt.LeftButton:
            return

        widget_pos = QPointF(event.position())
        img_pos = self.widget_to_image(widget_pos)

        if img_pos is None:
            return

        self.start_widget_point = widget_pos
        self.current_widget_point = widget_pos
        self.last_img_point = img_pos

        if self.crop_mode_enabled:
            self.mode = "crop"
            self.update_scaled_pixmap()
            return

        handle = self.main_window.get_handle_at(*img_pos)

        if handle is not None:
            self.main_window.push_undo_state()
            self.mode = "resizing"
            self.resize_handle = handle[1]
            self.main_window.selected_box_index = handle[0]
            self.original_box_before_edit = self.main_window.boxes[handle[0]].copy()
            self.update_scaled_pixmap()
            return

        border_box_index = self.main_window.get_border_at(*img_pos)

        if border_box_index is not None:
            self.main_window.push_undo_state()
            self.mode = "moving"
            self.main_window.selected_box_index = border_box_index
            self.original_box_before_edit = self.main_window.boxes[border_box_index].copy()
            self.update_scaled_pixmap()
            return

        clicked_box_index = self.main_window.find_box_at(*img_pos)
        self.main_window.selected_box_index = clicked_box_index

        self.mode = "drawing"
        self.resize_handle = None
        self.original_box_before_edit = None

        self.update_scaled_pixmap()

    def mouseMoveEvent(self, event):
        if self.main_window is None or self.base_pixmap is None:
            return

        widget_pos = QPointF(event.position())
        img_pos = self.widget_to_image(widget_pos)

        if self.mode in ("drawing", "crop"):
            self.current_widget_point = widget_pos
            self.update_scaled_pixmap()
            return

        if img_pos is None:
            return

        if self.mode == "moving":
            if self.main_window.selected_box_index is None or self.last_img_point is None:
                return

            dx = img_pos[0] - self.last_img_point[0]
            dy = img_pos[1] - self.last_img_point[1]

            box = self.main_window.boxes[self.main_window.selected_box_index]
            box.move(dx, dy, self.main_window.current_image_w, self.main_window.current_image_h)

            self.last_img_point = img_pos
            self.update_scaled_pixmap()
            self.main_window.update_status(extra="Moving box...")
            return

        if self.mode == "resizing":
            if self.main_window.selected_box_index is None:
                return

            box = self.main_window.boxes[self.main_window.selected_box_index]
            x, y = img_pos

            if self.resize_handle == "tl":
                box.x1, box.y1 = x, y
            elif self.resize_handle == "tr":
                box.x2, box.y1 = x, y
            elif self.resize_handle == "bl":
                box.x1, box.y2 = x, y
            elif self.resize_handle == "br":
                box.x2, box.y2 = x, y

            box.clamp(self.main_window.current_image_w, self.main_window.current_image_h)

            self.update_scaled_pixmap()
            self.main_window.update_status(extra="Resizing box...")
            return

    def mouseReleaseEvent(self, event):
        if self.base_pixmap is None or self.main_window is None:
            return

        if event.button() != Qt.LeftButton:
            return

        if self.mode == "crop":
            end_point = QPointF(event.position())

            start_img = self.widget_to_image(self.start_widget_point) if self.start_widget_point else None
            end_img = self.widget_to_image(end_point)

            self.mode = "idle"
            self.crop_mode_enabled = False

            if start_img is not None and end_img is not None:
                self.main_window.crop_current_image(start_img, end_img)
            else:
                self.main_window.update_status(extra="Crop cancelled.")

            self.start_widget_point = None
            self.current_widget_point = None
            self.last_img_point = None
            self.update_scaled_pixmap()
            return

        if self.mode == "drawing":
            end_point = QPointF(event.position())

            start_img = self.widget_to_image(self.start_widget_point) if self.start_widget_point else None
            end_img = self.widget_to_image(end_point)

            if start_img is not None and end_img is not None:
                x1, y1 = start_img
                x2, y2 = end_img

                new_box = Box(self.main_window.current_class_id, x1, y1, x2, y2)

                if not new_box.is_too_small():
                    self.main_window.push_undo_state()
                    self.main_window.boxes.append(new_box)
                    self.main_window.selected_box_index = len(self.main_window.boxes) - 1
                    self.main_window.update_status(extra="Box added.")
                else:
                    self.main_window.update_status(extra="Ignored tiny box.")

        elif self.mode in ("moving", "resizing"):
            if self.main_window.selected_box_index is not None:
                box = self.main_window.boxes[self.main_window.selected_box_index]
                box.clamp(self.main_window.current_image_w, self.main_window.current_image_h)

                if box.is_too_small():
                    if self.original_box_before_edit is not None:
                        self.main_window.boxes[self.main_window.selected_box_index] = self.original_box_before_edit
                    self.main_window.update_status(extra="Edit cancelled: box too small.")
                else:
                    self.main_window.update_status(extra="Box updated.")

        self.mode = "idle"
        self.resize_handle = None
        self.start_widget_point = None
        self.current_widget_point = None
        self.last_img_point = None
        self.original_box_before_edit = None

        self.update_scaled_pixmap()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Water Meter YOLO Labeler")
        self.resize(1280, 850)

        self.image_dir = None
        self.label_dir = None
        self.image_files = []
        self.current_index = 0

        self.current_image = None
        self.current_image_w = 0
        self.current_image_h = 0

        self.boxes = []
        self.selected_box_index = None
        self.current_class_id = 0
        self.undo_stack = []

        self.canvas = ImageCanvas()
        self.canvas.set_main_window(self)

        self.info_label = QLabel("Open an image folder to begin.")
        self.info_label.setWordWrap(True)

        open_btn = QPushButton("Open Folder")
        open_btn.clicked.connect(self.open_folder)

        prev_btn = QPushButton("Previous (P)")
        prev_btn.clicked.connect(self.prev_image)

        next_btn = QPushButton("Next (N)")
        next_btn.clicked.connect(self.next_image)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_labels)

        remove_btn = QPushButton("Remove Image")
        remove_btn.clicked.connect(self.remove_current_image)

        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self.undo)

        crop_btn = QPushButton("Crop Image")
        crop_btn.clicked.connect(self.enable_crop_mode)

        rot_left_btn = QPushButton("Rotate -15°")
        rot_left_btn.clicked.connect(lambda: self.rotate_current_image(-15))

        rot_right_btn = QPushButton("Rotate +15°")
        rot_right_btn.clicked.connect(lambda: self.rotate_current_image(15))

        rot_90_btn = QPushButton("Rotate 90°")
        rot_90_btn.clicked.connect(lambda: self.rotate_current_image(90))

        rot_custom_btn = QPushButton("Rotate Custom")
        rot_custom_btn.clicked.connect(self.rotate_custom_angle)

        top_bar = QHBoxLayout()
        top_bar.addWidget(open_btn)
        top_bar.addWidget(prev_btn)
        top_bar.addWidget(next_btn)
        top_bar.addWidget(save_btn)
        top_bar.addWidget(remove_btn)
        top_bar.addWidget(undo_btn)
        top_bar.addWidget(crop_btn)
        top_bar.addWidget(rot_left_btn)
        top_bar.addWidget(rot_right_btn)
        top_bar.addWidget(rot_90_btn)
        top_bar.addWidget(rot_custom_btn)
        top_bar.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_bar)
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(self.info_label)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        self.create_shortcuts()
        self.create_menu()
        self.update_status()

    def create_menu(self):
        open_action = QAction("Open Folder", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_folder)

        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_labels)

        remove_action = QAction("Remove Image", self)
        remove_action.setShortcut(QKeySequence("Ctrl+D"))
        remove_action.triggered.connect(self.remove_current_image)

        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo)

        crop_action = QAction("Crop Image", self)
        crop_action.setShortcut(QKeySequence("C"))
        crop_action.triggered.connect(self.enable_crop_mode)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addAction(remove_action)

        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction(undo_action)
        edit_menu.addAction(crop_action)

    def create_shortcuts(self):
        shortcuts = [
            ("OpenFolder", "Ctrl+O", self.open_folder),
            ("Save", "Ctrl+S", self.save_labels),
            ("Undo", "Ctrl+Z", self.undo),
            ("Next", "N", self.next_image),
            ("Prev", "P", self.prev_image),
            ("RemoveImage", "Ctrl+D", self.remove_current_image),
            ("CropImage", "C", self.enable_crop_mode),
            ("RotateMinus15", "Ctrl+Left", lambda: self.rotate_current_image(-15)),
            ("RotatePlus15", "Ctrl+Right", lambda: self.rotate_current_image(15)),
            ("Rotate90", "R", lambda: self.rotate_current_image(90)),
            ("RotateCustom", "Ctrl+R", self.rotate_custom_angle),
            ("Meter", "M", lambda: self.set_class(0)),
            ("Window", "W", lambda: self.set_class(1)),
            ("ZeroDigit", "0", lambda: self.set_class(2)),
            ("OneDigit", "1", lambda: self.set_class(3)),
            ("TwoDigit", "2", lambda: self.set_class(4)),
            ("ThreeDigit", "3", lambda: self.set_class(5)),
            ("FourDigit", "4", lambda: self.set_class(6)),
            ("FiveDigit", "5", lambda: self.set_class(7)),
            ("SixDigit", "6", lambda: self.set_class(8)),
            ("SevenDigit", "7", lambda: self.set_class(9)),
            ("EightDigit", "8", lambda: self.set_class(10)),
            ("NineDigit", "9", lambda: self.set_class(11)),
            ("UnknownDigit", "U", lambda: self.set_class(12)),
            ("Delete", "Delete", self.delete_selected_box),
            ("DeleteBackspace", "Backspace", self.delete_selected_box),
            ("ClearSelection", "Escape", self.clear_selection),
        ]

        for name, shortcut, slot in shortcuts:
            action = QAction(name, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(slot)
            self.addAction(action)

    def enable_crop_mode(self):
        if self.current_image is None:
            self.update_status(extra="Open an image first.")
            return

        self.canvas.crop_mode_enabled = True
        self.canvas.mode = "idle"
        self.selected_box_index = None
        self.canvas.update_scaled_pixmap()
        self.update_status(extra="Crop mode enabled. Draw a rectangle around the useful meter area.")

    def open_folder(self):
        start_folder = self.image_dir if self.image_dir else os.getcwd()

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            start_folder,
            QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog,
        )

        if not folder:
            return

        image_files = list_image_files(folder)

        if not image_files:
            safe_message(
                self,
                "No Images",
                "This folder has no supported images.\n\nPlease open the folder that directly contains the images.",
            )
            return

        self.image_dir = folder

        parent_dir = os.path.dirname(folder)
        self.label_dir = os.path.join(parent_dir, "labels")
        os.makedirs(self.label_dir, exist_ok=True)

        self.image_files = image_files
        self.current_index = 0

        self.load_current_image()
        self.update_status(extra=f"Images: {self.image_dir} | Labels: {self.label_dir}")

    def current_label_path(self):
        if not self.label_dir or not self.image_files:
            return None

        stem = os.path.splitext(self.image_files[self.current_index])[0]
        return os.path.join(self.label_dir, f"{stem}.txt")

    def push_undo_state(self):
        self.undo_stack.append([box.copy() for box in self.boxes])
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:
            self.update_status(extra="Nothing to undo.")
            return

        self.boxes = self.undo_stack.pop()
        self.selected_box_index = None
        self.canvas.update_scaled_pixmap()
        self.update_status(extra="Undid last action.")

    def clear_selection(self):
        self.selected_box_index = None
        self.canvas.crop_mode_enabled = False
        self.canvas.update_scaled_pixmap()
        self.update_status(extra="Selection cleared.")

    def load_current_image(self):
        if not self.image_dir or not self.image_files:
            return

        image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
        image = cv2.imread(image_path)

        if image is None:
            self.canvas.clear_image()
            self.current_image = None
            self.current_image_w = 0
            self.current_image_h = 0
            self.boxes = []
            self.selected_box_index = None
            self.update_status(extra=f"Failed to load image: {self.image_files[self.current_index]}")
            return

        self.current_image = image
        self.current_image_h, self.current_image_w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        qimage = QImage(
            rgb.data,
            self.current_image_w,
            self.current_image_h,
            rgb.strides[0],
            QImage.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(qimage.copy())

        self.undo_stack = []
        self.boxes = []
        self.selected_box_index = None
        self.canvas.crop_mode_enabled = False

        self.canvas.set_image(pixmap)
        self.load_labels()
        self.update_status()

    def load_labels(self):
        self.boxes = []
        self.selected_box_index = None

        label_path = self.current_label_path()

        if not label_path or not os.path.exists(label_path):
            self.canvas.update_scaled_pixmap()
            return

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])

                    if class_id < 0 or class_id >= len(CLASS_NAMES):
                        continue

                    box = Box.from_normalized(
                        class_id,
                        x_center,
                        y_center,
                        width,
                        height,
                        self.current_image_w,
                        self.current_image_h,
                    )

                    self.boxes.append(box)

        except Exception as e:
            self.update_status(extra=f"Label load error: {e}")

        self.canvas.update_scaled_pixmap()

    def save_labels(self):
        label_path = self.current_label_path()

        if not label_path:
            return

        try:
            with open(label_path, "w", encoding="utf-8") as f:
                for box in self.boxes:
                    x_center, y_center, width, height = box.normalized(
                        self.current_image_w,
                        self.current_image_h,
                    )

                    f.write(
                        f"{box.class_id} "
                        f"{x_center:.6f} "
                        f"{y_center:.6f} "
                        f"{width:.6f} "
                        f"{height:.6f}\n"
                    )

            self.update_status(extra="Saved.")

        except Exception as e:
            self.update_status(extra=f"Save error: {e}")

    def clear_label_file(self):
        label_path = self.current_label_path()

        if label_path and os.path.exists(label_path):
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("")

    def crop_current_image(self, start_img, end_img):
        if not self.image_dir or not self.image_files:
            return

        image_path = os.path.join(self.image_dir, self.image_files[self.current_index])

        if self.boxes:
            confirmed = safe_confirm(
                self,
                "Crop Image",
                (
                    "This image already has labels.\n\n"
                    "Cropping it will clear the current boxes because their positions will no longer match.\n\n"
                    "Continue?"
                ),
                yes_text="Yes, Crop",
                no_text="Cancel",
            )

            if not confirmed:
                self.update_status(extra="Cropping cancelled.")
                return

        image = cv2.imread(image_path)

        if image is None:
            self.update_status(extra=f"Could not read image: {image_path}")
            return

        x1, y1 = start_img
        x2, y2 = end_img

        cropped = crop_image(image, x1, y1, x2, y2)

        if cropped is None:
            self.update_status(extra="Invalid crop area.")
            return

        ok = cv2.imwrite(image_path, cropped)

        if not ok:
            self.update_status(extra=f"Could not save cropped image: {image_path}")
            return

        self.clear_label_file()
        self.load_current_image()
        self.update_status(extra="Image cropped and saved. Labels cleared.")

    def remove_current_image(self):
        if not self.image_dir or not self.image_files:
            return

        image_name = self.image_files[self.current_index]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = self.current_label_path()

        confirmed = safe_confirm(
            self,
            "Remove Image",
            (
                "Remove this image from the dataset?\n\n"
                f"{image_name}\n\n"
                "The image and its matching label file will be moved safely to:\n"
                "removed_images/ and removed_labels/"
            ),
            yes_text="Yes, Remove",
            no_text="Cancel",
        )

        if not confirmed:
            self.update_status(extra="Image removal cancelled.")
            return

        dataset_root = os.path.dirname(self.image_dir)

        removed_images_dir = os.path.join(dataset_root, "removed_images")
        removed_labels_dir = os.path.join(dataset_root, "removed_labels")

        os.makedirs(removed_images_dir, exist_ok=True)
        os.makedirs(removed_labels_dir, exist_ok=True)

        try:
            self.canvas.clear_image()
            self.current_image = None
            self.current_image_w = 0
            self.current_image_h = 0
            self.boxes = []
            self.selected_box_index = None
            self.undo_stack = []

            if os.path.exists(image_path):
                image_destination = unique_destination_path(os.path.join(removed_images_dir, image_name))
                shutil.move(image_path, image_destination)

            if label_path and os.path.exists(label_path):
                label_destination = unique_destination_path(os.path.join(removed_labels_dir, os.path.basename(label_path)))
                shutil.move(label_path, label_destination)

            del self.image_files[self.current_index]

            if not self.image_files:
                self.current_index = 0
                self.update_status(extra="Removed image. No images left.")
                return

            if self.current_index >= len(self.image_files):
                self.current_index = len(self.image_files) - 1

            self.load_current_image()
            self.update_status(extra=f"Removed image: {image_name}")

        except Exception as e:
            self.update_status(extra=f"Remove error: {e}")

    def rotate_current_image(self, angle_degrees: float):
        if not self.image_dir or not self.image_files:
            return

        image_path = os.path.join(self.image_dir, self.image_files[self.current_index])

        if self.boxes:
            confirmed = safe_confirm(
                self,
                "Rotate Image",
                (
                    "This image already has labels.\n\n"
                    "Rotating it will clear the current boxes because their positions will no longer match.\n\n"
                    "Continue?"
                ),
                yes_text="Yes, Rotate",
                no_text="Cancel",
            )

            if not confirmed:
                self.update_status(extra="Rotation cancelled.")
                return

        image = cv2.imread(image_path)

        if image is None:
            self.update_status(extra=f"Could not read image: {image_path}")
            return

        rotated = rotate_image_keep_size_crop_edges(image, angle_degrees)
        ok = cv2.imwrite(image_path, rotated)

        if not ok:
            self.update_status(extra=f"Could not save rotated image: {image_path}")
            return

        self.clear_label_file()
        self.load_current_image()
        self.update_status(extra=f"Image rotated by {angle_degrees}° and saved. Labels cleared.")

    def rotate_custom_angle(self):
        angle, ok = QInputDialog.getDouble(
            self,
            "Rotate Image",
            "Enter rotation angle in degrees:",
            0.0,
            -360.0,
            360.0,
            2,
        )

        if ok:
            self.rotate_current_image(angle)

    def next_image(self):
        if not self.image_files:
            return

        self.save_labels()

        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()

    def prev_image(self):
        if not self.image_files:
            return

        self.save_labels()

        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def set_class(self, class_id: int):
        self.current_class_id = class_id
        self.canvas.crop_mode_enabled = False

        if self.selected_box_index is not None and 0 <= self.selected_box_index < len(self.boxes):
            if self.boxes[self.selected_box_index].class_id != class_id:
                self.push_undo_state()
                self.boxes[self.selected_box_index].class_id = class_id
                self.canvas.update_scaled_pixmap()
                self.update_status(extra=f"Selected box changed to {CLASS_NAMES[class_id]}.")
                return

        self.update_status(extra=f"Current class: {CLASS_NAMES[class_id]}")

    def delete_selected_box(self):
        if self.selected_box_index is None:
            self.update_status(extra="No box selected.")
            return

        if 0 <= self.selected_box_index < len(self.boxes):
            self.push_undo_state()
            del self.boxes[self.selected_box_index]
            self.selected_box_index = None
            self.canvas.update_scaled_pixmap()
            self.update_status(extra="Box deleted.")

    def find_box_at(self, x, y):
        for i in range(len(self.boxes) - 1, -1, -1):
            if self.boxes[i].contains(x, y):
                return i
        return None

    def get_handle_at(self, x, y):
        tolerance = max(6, HANDLE_SIZE / max(self.canvas.scale, 1e-6))

        for i in range(len(self.boxes) - 1, -1, -1):
            box = self.boxes[i]
            x_min, y_min, x_max, y_max = box.rect()

            handles = {
                "tl": (x_min, y_min),
                "tr": (x_max, y_min),
                "bl": (x_min, y_max),
                "br": (x_max, y_max),
            }

            for name, (hx, hy) in handles.items():
                if abs(x - hx) <= tolerance and abs(y - hy) <= tolerance:
                    return i, name

        return None

    def get_border_at(self, x, y):
        tolerance = max(6, HANDLE_SIZE / max(self.canvas.scale, 1e-6))

        for i in range(len(self.boxes) - 1, -1, -1):
            box = self.boxes[i]
            x_min, y_min, x_max, y_max = box.rect()

            near_left = abs(x - x_min) <= tolerance and y_min <= y <= y_max
            near_right = abs(x - x_max) <= tolerance and y_min <= y <= y_max
            near_top = abs(y - y_min) <= tolerance and x_min <= x <= x_max
            near_bottom = abs(y - y_max) <= tolerance and x_min <= x <= x_max

            if near_left or near_right or near_top or near_bottom:
                return i

        return None

    def draw_boxes(self, painter: QPainter, scale: float, offset_x: int, offset_y: int):
        for i, box in enumerate(self.boxes):
            x_min, y_min, x_max, y_max = box.rect()

            x1 = x_min * scale + offset_x
            y1 = y_min * scale + offset_y
            x2 = x_max * scale + offset_x
            y2 = y_max * scale + offset_y

            color = CLASS_COLORS.get(box.class_id, QColor(255, 255, 255))
            pen_width = 3 if i == self.selected_box_index else 2

            painter.setPen(QPen(color, pen_width))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(QPointF(x1, y1), QPointF(x2, y2)))

            label_text = CLASS_NAMES[box.class_id]
            text_width = 22 if label_text in list("0123456789") else len(label_text) * 8 + 12
            text_height = 20
            label_top = max(0, int(y1) - text_height)

            painter.fillRect(int(x1), label_top, text_width, text_height, color)
            painter.setPen(Qt.black)
            painter.drawText(int(x1) + 5, label_top + 15, label_text)

            if i == self.selected_box_index:
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(QPen(Qt.black, 1))

                handles = [
                    QRectF(x1 - HANDLE_SIZE / 2, y1 - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE),
                    QRectF(x2 - HANDLE_SIZE / 2, y1 - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE),
                    QRectF(x1 - HANDLE_SIZE / 2, y2 - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE),
                    QRectF(x2 - HANDLE_SIZE / 2, y2 - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE),
                ]

                for handle_rect in handles:
                    painter.drawRect(handle_rect)

    def update_status(self, extra: str = ""):
        shortcuts_text = (
            "M=meter | W=window | 0..9=digit classes | U=unknown | "
            "C=crop image | N=next | P=previous | R=rotate 90° | "
            "Ctrl+Left=-15° | Ctrl+Right=+15° | Ctrl+R=custom rotate | "
            "Border drag=move box | Corner drag=resize | Inside click=select | "
            "Delete=delete box | Ctrl+D=remove image | Ctrl+Z=undo | Esc=clear selection"
        )

        if not self.image_files:
            self.info_label.setText(f"Open an image folder to begin.\n{shortcuts_text}")
            return

        image_name = self.image_files[self.current_index]
        class_name = CLASS_NAMES[self.current_class_id]

        selected_text = "None"

        if self.selected_box_index is not None and 0 <= self.selected_box_index < len(self.boxes):
            selected_text = f"{self.selected_box_index} ({CLASS_NAMES[self.boxes[self.selected_box_index].class_id]})"

        status = (
            f"Image {self.current_index + 1}/{len(self.image_files)}: {image_name} | "
            f"Current class: {self.current_class_id} ({class_name}) | "
            f"Boxes: {len(self.boxes)} | "
            f"Selected: {selected_text}"
        )

        if extra:
            status += f" | {extra}"

        status += "\n" + shortcuts_text

        self.info_label.setText(status)

    def closeEvent(self, event):
        try:
            self.save_labels()
        except Exception:
            pass

        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
