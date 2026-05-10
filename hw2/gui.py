"""
图像压缩对比 GUI
调用用户已实现的 img_decom.py (SVD) 和 dwt.py (DWT)
依赖: PyQt5, numpy, Pillow, pywt, matplotlib

目录结构（将本文件与 img_decom.py / dwt.py 放在同一文件夹）:
    your_project/
    ├── compression_gui.py   ← 本文件
    ├── img_decom.py         ← 你的 SVD 压缩模块
    └── dwt.py               ← 你的 DWT 压缩模块
"""

import sys
import os
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QFileDialog, QHBoxLayout, QVBoxLayout,
    QGroupBox, QComboBox, QSpinBox, QFrame,
    QSizePolicy, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont

# ══════════════════════════════════════════════
# 导入你写的两个模块
# ══════════════════════════════════════════════

try:
    from img_decom import compress_and_evaluate_image as svd_compress_and_evaluate
    SVD_AVAILABLE = True
    SVD_IMPORT_ERROR = ""
except ImportError as e:
    SVD_AVAILABLE = False
    SVD_IMPORT_ERROR = str(e)

try:
    from dwt import compress_and_evaluate_image as dwt_compress_and_evaluate
    DWT_AVAILABLE = True
    DWT_IMPORT_ERROR = ""
except ImportError as e:
    DWT_AVAILABLE = False
    DWT_IMPORT_ERROR = str(e)


# ══════════════════════════════════════════════
# 后台工作线程
# ══════════════════════════════════════════════

class CompressionWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, image_path, k, threshold, wavelet, level):
        super().__init__()
        self.image_path = image_path
        self.k          = k
        self.threshold  = threshold
        self.wavelet    = wavelet
        self.level      = level

    def run(self):
        result = {}
        try:
            # ── SVD（调用 img_decom.compress_and_evaluate_image）──────
            if SVD_AVAILABLE:
                svd_result = svd_compress_and_evaluate(
                    image_path=self.image_path,
                    k=self.k,
                    tol=1e-10,
                    save_path=None,
                    show=False,
                )
                # svd_result["compressed_image"] → uint8 ndarray
                # svd_result["metrics"]          → 完整 metrics dict
                result["svd_image"]   = svd_result["compressed_image"]
                result["svd_metrics"] = svd_result["metrics"]
            else:
                result["svd_error"] = SVD_IMPORT_ERROR

            # ── DWT（调用 dwt.compress_and_evaluate_image）────────────
            if DWT_AVAILABLE:
                dwt_result = dwt_compress_and_evaluate(
                    image_path=self.image_path,
                    threshold=self.threshold,
                    wavelet=self.wavelet,
                    level=self.level,
                    save_path=None,
                    show=False,
                )
                # dwt_result["compressed_image"] → uint8 ndarray
                # dwt_result["metrics"]          → 完整 metrics dict
                result["dwt_image"]   = dwt_result["compressed_image"]
                result["dwt_metrics"] = dwt_result["metrics"]
            else:
                result["dwt_error"] = DWT_IMPORT_ERROR

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


# ══════════════════════════════════════════════
# 图像显示控件
# ══════════════════════════════════════════════

class ImageLabel(QWidget):
    """
    用 paintEvent 手动绘制，确保图像始终铺满控件、保持宽高比居中，
    不依赖 QLabel.setPixmap 的尺寸缓存问题。
    """
    def __init__(self, placeholder=""):
        super().__init__()
        self.placeholder = placeholder
        self._pixmap = None
        self.setMinimumSize(220, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_ndarray(self, arr: np.ndarray):
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        h, w = arr.shape[:2]
        # 必须 copy()，防止 numpy buffer 被回收后 QImage 变花
        qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()   # 触发 paintEvent

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        rect = self.rect()

        if self._pixmap is None:
            # 绘制占位文字
            painter.setPen(QColor("#3a3a5a"))
            painter.setFont(QFont("Consolas", 13))
            painter.drawText(rect, Qt.AlignCenter, f"[ {self.placeholder} ]")
        else:
            # 按比例缩放后居中绘制
            scaled = self._pixmap.scaled(
                rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (rect.width()  - scaled.width())  // 2
            y = (rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        painter.end()


# ══════════════════════════════════════════════
# 指标卡片
# ══════════════════════════════════════════════

class MetricsPanel(QFrame):
    BASE_ROWS = [
        ("MSE",    "mse",               "{:.2f}"),
        ("PSNR",   "psnr",              "{:.2f} dB"),
        ("SSIM",   "ssim",              "{:.4f}"),
        ("压缩比", "compression_ratio", "{:.2f}×"),
    ]
    SVD_EXTRA = [
        ("能量(R)", "energy_ratio_R",   "{:.2%}"),
        ("能量(G)", "energy_ratio_G",   "{:.2%}"),
        ("能量(B)", "energy_ratio_B",   "{:.2%}"),
    ]
    DWT_EXTRA = [
        ("稀疏度",  "sparsity",         "{:.2%}"),
        ("能量保留","energy_ratio_avg", "{:.2%}"),
    ]

    def __init__(self, title, accent, extra_rows=None):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            MetricsPanel {{
                background: #14142a;
                border: 1px solid {accent}44;
                border-radius: 8px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(3)

        hdr = QLabel(title)
        hdr.setStyleSheet(
            f"color:{accent}; font-weight:bold; font-size:18px; font-family:'Consolas';")
        lay.addWidget(hdr)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{accent}33;")
        lay.addWidget(sep)

        self._vals = {}
        for name, key, fmt in self.BASE_ROWS + (extra_rows or []):
            row = QHBoxLayout()
            l_n = QLabel(name)
            l_n.setStyleSheet("color:#666; font-size:17px; font-family:'Consolas';")
            l_v = QLabel("—")
            l_v.setAlignment(Qt.AlignRight)
            l_v.setStyleSheet("color:#ccc; font-size:17px; font-family:'Consolas';")
            row.addWidget(l_n); row.addStretch(); row.addWidget(l_v)
            lay.addLayout(row)
            self._vals[key] = (l_v, fmt)

    def update_metrics(self, m: dict):
        for key, (lbl, fmt) in self._vals.items():
            if key in m:
                try:    lbl.setText(fmt.format(m[key]))
                except: lbl.setText(str(m[key]))
            else:
                lbl.setText("—")

    def show_error(self):
        for lbl, _ in self._vals.values():
            lbl.setText("N/A")


# ══════════════════════════════════════════════
# 主窗口
# ══════════════════════════════════════════════

ACCENT_SVD = "#00d4ff"
ACCENT_DWT = "#ff6b35"
BG_DARK    = "#0d0d1a"
BG_PANEL   = "#12122a"
BG_CARD    = "#1a1a2e"


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像压缩对比  ·  SVD  vs  DWT")
        self.resize(1440, 880)
        self._apply_theme()

        self.image_path = None
        self.worker     = None

        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._run)

        self._build_ui()
        self._check_modules()

    # ── 样式 ──────────────────────────────────

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background:{BG_DARK}; color:#ddd;
                font-family:'Segoe UI','Microsoft YaHei',sans-serif;
            }}
            QGroupBox {{
                border:1px solid #252548; border-radius:6px;
                margin-top:10px; color:#666; font-size:17px; padding:8px;
            }}
            QGroupBox::title {{ subcontrol-origin:margin; left:10px; }}
            QPushButton {{
                background:#1c1c3a; color:#bbc; border:1px solid #353565;
                border-radius:5px; padding:6px 16px; font-size:18px;
            }}
            QPushButton:hover  {{ background:#252550; border-color:{ACCENT_SVD}; color:#fff; }}
            QPushButton:pressed {{ background:#101025; }}
            QSlider::groove:horizontal {{
                background:#252545; height:4px; border-radius:2px;
            }}
            QSlider::handle:horizontal {{
                background:{ACCENT_SVD}; width:14px; height:14px;
                margin:-5px 0; border-radius:7px;
            }}
            QSlider::sub-page:horizontal {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {ACCENT_SVD},stop:1 {ACCENT_DWT});
                border-radius:2px;
            }}
            QComboBox, QSpinBox {{
                background:#18183a; color:#ccc; border:1px solid #353565;
                border-radius:4px; padding:4px 8px; font-size:17px;
            }}
            QStatusBar {{ color:#666; font-size:17px; }}
            QProgressBar {{
                border:none; background:#18183a; border-radius:2px; color:transparent;
            }}
            QProgressBar::chunk {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {ACCENT_SVD},stop:1 {ACCENT_DWT});
                border-radius:2px;
            }}
        """)

    # ── UI ────────────────────────────────────

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        vlay = QVBoxLayout(root)
        vlay.setSpacing(0); vlay.setContentsMargins(0,0,0,0)

        vlay.addWidget(self._mk_topbar())

        body = QHBoxLayout()
        body.setSpacing(0); body.setContentsMargins(0,0,0,0)
        body.addWidget(self._mk_sidebar(), 0)
        body.addWidget(self._mk_canvas(),  1)
        vlay.addLayout(body, 1)

        self.progress = QProgressBar()
        self.progress.setFixedHeight(3)
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        vlay.addWidget(self.progress)

    def _mk_topbar(self):
        bar = QWidget()
        bar.setFixedHeight(70)
        bar.setStyleSheet(f"background:{BG_PANEL}; border-bottom:1px solid #1e1e40;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 0, 20, 0)

        title = QLabel("IMAGE COMPRESSION LAB")
        title.setStyleSheet(f"""
            font-size:26px; font-weight:bold;
            font-family:'Courier New',monospace; letter-spacing:3px;
            color:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 {ACCENT_SVD},stop:1 {ACCENT_DWT});
        """)
        sub = QLabel("img_decom.py  ·  dwt.py  对比工具")
        sub.setStyleSheet(f"font-size:17px; color:#444; font-family:'Consolas';")

        self.btn_open = QPushButton("📂  选择图像")
        self.btn_open.setFixedHeight(36)
        self.btn_open.setStyleSheet(f"""
            QPushButton {{
                background:transparent; border:1px solid {ACCENT_SVD};
                color:{ACCENT_SVD}; border-radius:5px; padding:0 18px; font-size:18px;
            }}
            QPushButton:hover {{ background:rgba(0,212,255,0.1); }}
        """)
        self.btn_open.clicked.connect(self._open_image)

        lay.addWidget(title); lay.addSpacing(14); lay.addWidget(sub)
        lay.addStretch(); lay.addWidget(self.btn_open)
        return bar

    def _mk_sidebar(self):
        side = QWidget()
        side.setFixedWidth(350)
        side.setStyleSheet(f"background:{BG_PANEL}; border-right:1px solid #1e1e40;")
        lay = QVBoxLayout(side)
        lay.setContentsMargins(14, 16, 14, 16)
        lay.setSpacing(12)

        # SVD 参数
        svd_grp = QGroupBox("SVD 参数  (img_decom.py)")
        svd_grp.setStyleSheet(
            f"QGroupBox{{border-color:{ACCENT_SVD}44; color:{ACCENT_SVD};}}")
        sg = QVBoxLayout(svd_grp)

        k_hdr = QHBoxLayout()
        k_hdr.addWidget(QLabel("k ="))
        k_hdr.addStretch()
        self.lbl_k = QLabel("50")
        self.lbl_k.setStyleSheet(
            f"color:{ACCENT_SVD}; font-weight:bold; font-family:'Consolas'; font-size:19px;")
        k_hdr.addWidget(self.lbl_k)
        sg.addLayout(k_hdr)

        self.slider_k = QSlider(Qt.Horizontal)
        self.slider_k.setRange(1, 200)
        self.slider_k.setValue(50)
        self.slider_k.valueChanged.connect(self._on_k)
        sg.addWidget(self.slider_k)

        rng = QHBoxLayout()
        for t, a in [("1", Qt.AlignLeft), ("200", Qt.AlignRight)]:
            lb = QLabel(t); lb.setAlignment(a)
            lb.setStyleSheet("color:#2a2a4a; font-size:16px;")
            rng.addWidget(lb)
        sg.addLayout(rng)

        if not SVD_AVAILABLE:
            w = QLabel(f"⚠ img_decom.py 导入失败\n{SVD_IMPORT_ERROR}")
            w.setStyleSheet("color:#ff6666; font-size:14px;")
            w.setWordWrap(True)
            sg.addWidget(w)
            svd_grp.setEnabled(False)

        lay.addWidget(svd_grp)

        # DWT 参数
        dwt_grp = QGroupBox("DWT 参数  (dwt.py)")
        dwt_grp.setStyleSheet(
            f"QGroupBox{{border-color:{ACCENT_DWT}44; color:{ACCENT_DWT};}}")
        dg = QVBoxLayout(dwt_grp)
        dg.setSpacing(8)

        thr_hdr = QHBoxLayout()
        thr_hdr.addWidget(QLabel("threshold ="))
        thr_hdr.addStretch()
        self.lbl_thr = QLabel("10.0")
        self.lbl_thr.setStyleSheet(
            f"color:{ACCENT_DWT}; font-weight:bold; font-family:'Consolas'; font-size:19px;")
        thr_hdr.addWidget(self.lbl_thr)
        dg.addLayout(thr_hdr)

        self.slider_thr = QSlider(Qt.Horizontal)
        self.slider_thr.setRange(1, 1000)   # ×0.1 → 0.1 ~ 100
        self.slider_thr.setValue(100)
        self.slider_thr.setStyleSheet(f"""
            QSlider::handle:horizontal {{
                background:{ACCENT_DWT}; width:14px; height:14px;
                margin:-5px 0; border-radius:7px;
            }}
            QSlider::groove:horizontal {{
                background:#252545; height:4px; border-radius:2px;
            }}
            QSlider::sub-page:horizontal {{
                background:{ACCENT_DWT}; border-radius:2px;
            }}
        """)
        self.slider_thr.valueChanged.connect(self._on_thr)
        dg.addWidget(self.slider_thr)

        trng = QHBoxLayout()
        for t, a in [("0.1", Qt.AlignLeft), ("100", Qt.AlignRight)]:
            lb = QLabel(t); lb.setAlignment(a)
            lb.setStyleSheet("color:#2a2a4a; font-size:16px;")
            trng.addWidget(lb)
        dg.addLayout(trng)

        wv_row = QHBoxLayout()
        wv_row.addWidget(QLabel("wavelet"))
        self.combo_wv = QComboBox()
        self.combo_wv.addItems(["haar","db1","db2","db4","db8","sym4","coif2","bior2.2"])
        self.combo_wv.currentTextChanged.connect(self._on_param_change)
        wv_row.addWidget(self.combo_wv)
        dg.addLayout(wv_row)

        lv_row = QHBoxLayout()
        lv_row.addWidget(QLabel("level"))
        self.spin_lv = QSpinBox()
        self.spin_lv.setRange(1, 6)
        self.spin_lv.setValue(1)
        self.spin_lv.valueChanged.connect(self._on_param_change)
        lv_row.addWidget(self.spin_lv)
        dg.addLayout(lv_row)

        if not DWT_AVAILABLE:
            w = QLabel(f"⚠ dwt.py 导入失败\n{DWT_IMPORT_ERROR}")
            w.setStyleSheet("color:#ff6666; font-size:14px;")
            w.setWordWrap(True)
            dg.addWidget(w)
            dwt_grp.setEnabled(False)

        lay.addWidget(dwt_grp)

        # 指标面板
        self.panel_svd = MetricsPanel(
            "SVD 指标", ACCENT_SVD, extra_rows=MetricsPanel.SVD_EXTRA)
        self.panel_dwt = MetricsPanel("DWT 指标", ACCENT_DWT, extra_rows=MetricsPanel.DWT_EXTRA)
        lay.addWidget(self.panel_svd)
        lay.addWidget(self.panel_dwt)

        lay.addStretch()

        self.lbl_info = QLabel("未加载图像")
        self.lbl_info.setStyleSheet(
            "color:#3a3a5a; font-size:16px; font-family:'Consolas';")
        self.lbl_info.setWordWrap(True)
        lay.addWidget(self.lbl_info)

        return side

    def _mk_canvas(self):
        area = QWidget()
        area.setStyleSheet(f"background:{BG_DARK};")
        lay = QVBoxLayout(area)
        lay.setContentsMargins(16, 14, 16, 10)
        lay.setSpacing(10)

        hdr = QHBoxLayout()
        for txt, col in [
            ("原  图",                    "#555"),
            (f"SVD  (img_decom.py)",      ACCENT_SVD),
            (f"DWT  (dwt.py)",            ACCENT_DWT),
        ]:
            lb = QLabel(txt); lb.setAlignment(Qt.AlignCenter)
            lb.setStyleSheet(
                f"color:{col}; font-size:19px; font-weight:bold;"
                f"font-family:'Consolas'; letter-spacing:1px;")
            hdr.addWidget(lb, 1)
        lay.addLayout(hdr)

        imgs = QHBoxLayout(); imgs.setSpacing(12)
        self.img_orig = ImageLabel("原图")
        self.img_svd  = ImageLabel("SVD")
        self.img_dwt  = ImageLabel("DWT")

        for w in (self.img_orig, self.img_svd, self.img_dwt):
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{
                    background:{BG_CARD};
                    border:1px solid #1e1e40;
                    border-radius:8px;
                }}
            """)
            cl = QVBoxLayout(card); cl.setContentsMargins(4, 4, 4, 4)
            cl.addWidget(w)
            imgs.addWidget(card, 1)

        lay.addLayout(imgs, 1)
        return area

    # ── 模块状态检查 ──────────────────────────

    def _check_modules(self):
        msgs = []
        if not SVD_AVAILABLE:
            msgs.append(f"⚠ img_decom.py：{SVD_IMPORT_ERROR}")
        if not DWT_AVAILABLE:
            msgs.append(f"⚠ dwt.py：{DWT_IMPORT_ERROR}")
        if msgs:
            self.statusBar().showMessage("  |  ".join(msgs))
        else:
            self.statusBar().showMessage(
                "✓ img_decom.py 与 dwt.py 均已加载  ·  请选择图像开始")

    # ── 事件处理 ──────────────────────────────

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)")
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self.image_path = path
            arr = np.array(img, dtype=np.uint8)
            self.img_orig.set_ndarray(arr)
            h, w = arr.shape[:2]
            self.lbl_info.setText(
                f"{os.path.basename(path)}\n{w} × {h} px  |  RGB")

            # 动态限制 k 上限
            max_k = min(h, w, 300)
            self.slider_k.setRange(1, max_k)
            if self.slider_k.value() > max_k:
                self.slider_k.setValue(min(50, max_k))

            self._schedule()
        except Exception as e:
            self.statusBar().showMessage(f"加载失败: {e}")

    def _on_k(self, v):
        self.lbl_k.setText(str(v))
        self._schedule()

    def _on_thr(self, v):
        self.lbl_thr.setText(f"{v * 0.1:.1f}")
        self._schedule()

    def _on_param_change(self):
        self._schedule()

    def _schedule(self):
        if self.image_path:
            self._debounce.start(350)

    def _run(self):
        if not self.image_path:
            return
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()

        k         = self.slider_k.value()
        threshold = self.slider_thr.value() * 0.1
        wavelet   = self.combo_wv.currentText()
        level     = self.spin_lv.value()

        self.progress.setVisible(True)
        self.statusBar().showMessage(
            f"计算中 …  SVD k={k}  |  DWT threshold={threshold:.1f}"
            f"  wavelet={wavelet}  level={level}")

        self.worker = CompressionWorker(
            self.image_path, k, threshold, wavelet, level)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_err)
        self.worker.start()

    def _on_done(self, result):
        self.progress.setVisible(False)

        if "svd_image" in result:
            self.img_svd.set_ndarray(result["svd_image"])
            self.panel_svd.update_metrics(result["svd_metrics"])
        else:
            self.panel_svd.show_error()

        if "dwt_image" in result:
            self.img_dwt.set_ndarray(result["dwt_image"])
            self.panel_dwt.update_metrics(result["dwt_metrics"])
        else:
            self.panel_dwt.show_error()

        sm = result.get("svd_metrics", {})
        dm = result.get("dwt_metrics", {})
        self.statusBar().showMessage(
            f"SVD  PSNR={sm.get('psnr', 0):.1f}dB  "
            f"SSIM={sm.get('ssim', 0):.4f}  "
            f"CR={sm.get('compression_ratio', 0):.2f}×"
            f"    ║    "
            f"DWT  PSNR={dm.get('psnr', 0):.1f}dB  "
            f"SSIM={dm.get('ssim', 0):.4f}  "
            f"CR={dm.get('compression_ratio', 0):.2f}×"
        )

    def _on_err(self, msg):
        self.progress.setVisible(False)
        self.statusBar().showMessage(f"计算出错: {msg}")


# ══════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
