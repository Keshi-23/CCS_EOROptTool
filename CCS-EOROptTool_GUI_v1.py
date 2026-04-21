"""
CCS-EOROptTool_GUI_v1.py
========================
Patched GUI with sliders and live (auto-updating) prediction.

Differences vs the original CCS-EOROptTool_GUI.py
-------------------------------------------------
1. Each of the nine input parameters now has a slider AND a spin box.
   Moving the slider updates the spin box (and vice versa) and triggers
   an automatic recomputation of CO2 storage and cumulative oil.
2. Predictions are recomputed live (debounced 80 ms) so you can scrub
   sliders and see the response surface in real time.
3. The predicted CO2 / Oil values are shown at the top of the Input tab,
   above the parameter form, so you don't have to switch tabs to see
   them while moving sliders.
4. Slider min/max defaults match Table 1 of the manuscript (the training
   envelope of the published ANN). The model loaders bound them to the
   training X_min / X_max read from the JSON files.
5. JSON loader is unchanged and remains binary-compatible with files
   produced by train_ann_ccs_eor.py and with the original MATLAB exports.
6. Fixed an existing bug in _on_clear_clicked (sowv -> sorw).

Drop this file in the same folder as TrainedNN_FullCO2.json and
TrainedNN_FullOil.json. Run with:

    python CCS-EOROptTool_GUI_v1.py
"""

import sys
import os
import json
import numpy as np
from scipy.stats import norm  # for inverse normal CDF used in LHS

from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QBrush
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
    QGroupBox,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QScrollArea,
    QFrame,
    QMessageBox,
    QStatusBar,
    QSlider,
    QComboBox,
)


# =============================================================================
# Styling (unchanged)
# =============================================================================
def load_stylesheet() -> str:
    blue_900 = "#0f1d3c"
    blue_800 = "#13254d"
    blue_600 = "#1f3b73"
    teal_accent = "#2fb4c8"
    text_primary = "#e6eefc"
    text_muted = "#b8c3e0"

    return f"""
        * {{
            font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            font-size: 18px;
        }}
        QMainWindow {{ background: {blue_900}; }}
        QAbstractScrollArea {{ background: {blue_900}; border: 0px; }}
        QScrollArea {{ border: 0px; }}
        QWidget#body {{ background: {blue_900}; }}

        QTabWidget::pane {{ border: 0px; background: {blue_900}; }}
        QTabBar::tab {{
            background: {blue_800};
            color: {text_muted};
            padding: 18px 36px;
            margin-right: 8px;
            border-top-left-radius: 12px;
            border-top-right-radius: 12px;
            font-size: 22px;
            min-width: 200px;
            min-height: 50px;
        }}
        QTabBar::tab:selected {{
            background: {blue_600};
            color: {text_primary};
        }}

        QLabel {{ color: {text_primary}; font-size: 20px; }}
        QLabel#colHeader {{
            color: {text_primary};
            font-size: 20px;
            font-weight: 800;
        }}
        QLabel#liveValue {{
            color: {teal_accent};
            font-size: 28px;
            font-weight: 800;
        }}
        QLabel#liveLabel {{
            color: {text_primary};
            font-size: 18px;
        }}

        QGroupBox {{
            color: {text_primary};
            background: {blue_800};
            border: 1px solid {blue_600};
            border-radius: 14px;
            margin-top: 0px;
            padding: 8px 14px 14px 14px;
            font-weight: 600;
        }}

        QDoubleSpinBox {{
            background: {blue_900};
            color: {text_primary};
            border: 1px solid {blue_600};
            border-radius: 10px;
            padding: 6px 8px;
            font-size: 18px;
            min-width: 140px;
        }}

        QSpinBox {{
            background: {blue_900};
            color: {text_primary};
            border: 1px solid {blue_600};
            border-radius: 10px;
            padding: 6px 8px;
            font-size: 18px;
            min-width: 100px;
        }}

        QComboBox {{
            background: {blue_900};
            color: {text_primary};
            border: 1px solid {blue_600};
            border-radius: 10px;
            padding: 6px 12px;
            font-size: 18px;
            min-width: 200px;
        }}
        QComboBox::drop-down {{
            border: 0px;
            width: 28px;
        }}
        QComboBox QAbstractItemView {{
            background: {blue_800};
            color: {text_primary};
            border: 1px solid {blue_600};
            selection-background-color: {teal_accent};
            selection-color: #08131f;
        }}

        QDoubleSpinBox#cvBox {{
            min-width: 80px;
            max-width: 100px;
            padding: 4px 6px;
            font-size: 16px;
        }}

        QLabel#uqStatValue {{
            color: {teal_accent};
            font-size: 20px;
            font-weight: 700;
        }}
        QLabel#uqStatLabel {{
            color: {text_primary};
            font-size: 18px;
        }}

        QPushButton {{
            background: {teal_accent};
            color: #08131f;
            border: 0px;
            border-radius: 12px;
            padding: 12px 24px;
            font-weight: 700;
            min-width: 140px;
        }}
        QPushButton:hover {{ background: #3fc7da; }}

        QSlider::groove:horizontal {{
            border: 1px solid {blue_600};
            height: 8px;
            background: {blue_900};
            margin: 0;
            border-radius: 4px;
        }}
        QSlider::handle:horizontal {{
            background: {teal_accent};
            border: 1px solid {teal_accent};
            width: 22px;
            height: 22px;
            margin: -8px 0;
            border-radius: 11px;
        }}
        QSlider::sub-page:horizontal {{
            background: {teal_accent};
            border-radius: 4px;
        }}
        QStatusBar {{ color: {text_muted}; }}
    """


# =============================================================================
# Prediction engine -- JSON format unchanged from the original GUI
# =============================================================================
class PredictionEngine:
    FEATURE_ORDER = [
        "BHP_psi",
        "Area_ft2",
        "InjRate_ft3_day",
        "POR",
        "Perm_mD",
        "Thickness_ft",
        "Depth_ft",
        "S_org",
        "S_orw",
    ]

    def __init__(self, model_dir: str = None):
        try:
            base = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base = os.getcwd()
        self.model_dir = model_dir or base
        self.co2_model = None
        self.oil_model = None
        self.co2_model_path = None
        self.oil_model_path = None

    def ensure_loaded(self):
        if self.co2_model is None:
            self._autoload("co2")
        if self.oil_model is None:
            self._autoload("oil")

    def predict(self, values: dict) -> tuple:
        self.ensure_loaded()
        X = np.array([[values[k] for k in self.FEATURE_ORDER]], dtype=float)
        co2_norm = self._predict_json(self.co2_model, X)
        oil_norm = self._predict_json(self.oil_model, X)
        # Step 1: denormalize from [0,1] back to the model's target space
        # (which may itself be transformed, e.g. log10).
        co2_t = co2_norm * (self.co2_model["co2_max"] - self.co2_model["co2_min"]) + self.co2_model["co2_min"]
        oil_t = oil_norm * (self.oil_model["oil_max"] - self.oil_model["oil_min"]) + self.oil_model["oil_min"]
        # Step 2: apply the inverse of any forward transform recorded in the
        # JSON metadata to recover the prediction in original physical units.
        co2_real = self._inv_transform(co2_t, self.co2_model["target_transform"])
        oil_real = self._inv_transform(oil_t, self.oil_model["target_transform"])
        return max(0.0, float(co2_real)), max(0.0, float(oil_real))

    @staticmethod
    def _inv_transform(y: float, kind: str) -> float:
        """Mirror of train_ann_ccs_eor._inv_transform.

        Supported transforms:
            "identity"  : y
            "log10p1"   : 10**y - 1     (inverse of log10(y + 1))
        """
        kind = (kind or "identity").lower()
        if kind == "identity":
            return float(y)
        if kind == "log10p1":
            return float(10.0 ** y - 1.0)
        raise ValueError(f"Unknown target_transform in JSON: {kind!r}")

    def _autoload(self, which: str):
        want = "TrainedNN_FullCO2.json" if which == "co2" else "TrainedNN_FullOil.json"
        path = os.path.join(self.model_dir, want)
        if not os.path.isfile(path):
            # Last-resort scan
            for f in os.listdir(self.model_dir):
                if f.lower().endswith(".json") and which in f.lower():
                    path = os.path.join(self.model_dir, f)
                    break
            else:
                raise FileNotFoundError(
                    f"Could not find {want} in {self.model_dir}"
                )
        with open(path, "r") as f:
            nn = json.load(f)

        iw = np.asarray(nn["IW"], dtype=float)
        lw = np.asarray(nn["LW"], dtype=float).reshape(1, -1)
        b = nn["b"]
        b1 = np.asarray(b[0], dtype=float).reshape(-1)
        b2 = float(np.asarray(b[1], dtype=float))
        x_min = np.asarray(nn["X_min"], dtype=float).reshape(-1)
        x_max = np.asarray(nn["X_max"], dtype=float).reshape(-1)

        if iw.shape[1] != len(self.FEATURE_ORDER):
            raise ValueError(
                f"Model expects {iw.shape[1]} features but GUI has {len(self.FEATURE_ORDER)}"
            )

        model = {
            "IW": iw, "LW": lw, "b1": b1, "b2": b2,
            "X_min": x_min, "X_max": x_max,
            "co2_min": float(nn.get("CO2_min", 0)),
            "co2_max": float(nn.get("CO2_max", 1)),
            "oil_min": float(nn.get("Oil_min", 0)),
            "oil_max": float(nn.get("Oil_max", 1)),
            # Optional inverse-transform metadata. Default "identity" keeps
            # backwards compatibility with all older JSON files (MATLAB
            # exports and earlier sklearn-trained ones).
            "target_transform": str(nn.get("target_transform", "identity")).lower(),
        }
        if which == "co2":
            self.co2_model, self.co2_model_path = model, path
        else:
            self.oil_model, self.oil_model_path = model, path

    @staticmethod
    def _predict_json(model: dict, X: np.ndarray) -> float:
        x_vec = X.reshape(-1)
        denom = model["X_max"] - model["X_min"]
        denom = np.where(denom == 0, 1.0, denom)
        x_norm = (x_vec - model["X_min"]) / denom
        h = np.tanh(np.dot(model["IW"], x_norm) + model["b1"])
        y = np.dot(model["LW"], h) + model["b2"]
        return float(np.ravel(y)[0])

    # ----- Vectorized batch path used by the Uncertainty tab ------------
    @staticmethod
    def _predict_json_batch(model: dict, X: np.ndarray) -> np.ndarray:
        """Vectorized forward pass.

        X: shape (N, 9) in original (un-normalized) units.
        Returns y_norm: shape (N,) in [0, 1] normalized space.
        """
        denom = model["X_max"] - model["X_min"]
        denom = np.where(denom == 0, 1.0, denom)
        Xn = (X - model["X_min"]) / denom                       # (N, 9)
        H = np.tanh(Xn @ model["IW"].T + model["b1"])           # (N, H)
        Y = H @ model["LW"].T + model["b2"]                     # (N, 1)
        return Y.reshape(-1)

    @staticmethod
    def _inv_transform_array(y: np.ndarray, kind: str) -> np.ndarray:
        kind = (kind or "identity").lower()
        if kind == "identity":
            return y
        if kind == "log10p1":
            return np.power(10.0, y) - 1.0
        raise ValueError(f"Unknown target_transform: {kind!r}")

    def predict_batch(self, X: np.ndarray) -> tuple:
        """X shape (N, 9). Returns (co2_array, oil_array) in original units."""
        self.ensure_loaded()
        co2_norm = self._predict_json_batch(self.co2_model, X)
        oil_norm = self._predict_json_batch(self.oil_model, X)
        co2_t = co2_norm * (self.co2_model["co2_max"] - self.co2_model["co2_min"]) + self.co2_model["co2_min"]
        oil_t = oil_norm * (self.oil_model["oil_max"] - self.oil_model["oil_min"]) + self.oil_model["oil_min"]
        co2 = self._inv_transform_array(co2_t, self.co2_model["target_transform"])
        oil = self._inv_transform_array(oil_t, self.oil_model["target_transform"])
        return np.maximum(0.0, co2), np.maximum(0.0, oil)

    def envelope(self) -> tuple:
        """Returns (X_min, X_max) from the loaded CO2 model (= Oil model)."""
        self.ensure_loaded()
        return (self.co2_model["X_min"].copy(), self.co2_model["X_max"].copy())


# =============================================================================
# Slider + spin-box compound widget (one row per parameter)
# =============================================================================
class SliderRow:
    """Bundles a label, a horizontal slider, and a spin box.

    The slider is integer-only (resolution = 10,000 steps) and is mapped
    linearly to a continuous parameter range. Slider and spin box stay in
    sync without recursive signal loops.
    """
    SLIDER_RES = 10000

    def __init__(self, label_text: str, vmin: float, vmax: float,
                 default: float, decimals: int, step: float,
                 on_change=None):
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.decimals = decimals
        self.on_change = on_change

        self.label = QLabel(label_text)
        self.label.setMinimumWidth(180)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.SLIDER_RES)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(max(1, self.SLIDER_RES // 50))
        self.slider.setMinimumWidth(280)

        self.box = QDoubleSpinBox()
        self.box.setRange(vmin, vmax)
        self.box.setDecimals(decimals)
        self.box.setSingleStep(step)
        self.box.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.box.setMinimumHeight(40)
        self.box.setMinimumWidth(150)

        # Wire up bidirectional sync
        self.slider.valueChanged.connect(self._slider_changed)
        self.box.valueChanged.connect(self._box_changed)

        # Initial value
        self.set_value(default)

    # -- public --
    def value(self) -> float:
        return float(self.box.value())

    def set_value(self, v: float):
        v = max(self.vmin, min(self.vmax, float(v)))
        self.box.blockSignals(True)
        self.slider.blockSignals(True)
        self.box.setValue(v)
        self.slider.setValue(self._to_slider(v))
        self.box.blockSignals(False)
        self.slider.blockSignals(False)

    def set_range(self, vmin: float, vmax: float):
        """Re-bind the parameter range (used after model JSON loads)."""
        cur = self.value()
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.box.blockSignals(True)
        self.slider.blockSignals(True)
        self.box.setRange(self.vmin, self.vmax)
        # Recompute slider from current value within new range
        cur = max(self.vmin, min(self.vmax, cur))
        self.box.setValue(cur)
        self.slider.setValue(self._to_slider(cur))
        self.box.blockSignals(False)
        self.slider.blockSignals(False)

    # -- internal --
    def _to_slider(self, v: float) -> int:
        if self.vmax == self.vmin:
            return 0
        frac = (v - self.vmin) / (self.vmax - self.vmin)
        return int(round(frac * self.SLIDER_RES))

    def _from_slider(self, s: int) -> float:
        frac = s / self.SLIDER_RES
        return self.vmin + frac * (self.vmax - self.vmin)

    def _slider_changed(self, s: int):
        v = self._from_slider(s)
        self.box.blockSignals(True)
        self.box.setValue(v)
        self.box.blockSignals(False)
        if self.on_change is not None:
            self.on_change()

    def _box_changed(self, v: float):
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_slider(v))
        self.slider.blockSignals(False)
        if self.on_change is not None:
            self.on_change()


# =============================================================================
# Input tab with sliders + live result panel at the top
# =============================================================================
class InputTab(QWidget):
    # Defaults pulled from Table 1 of the manuscript (training envelope)
    PARAM_DEFAULTS = [
        # key,        label,                   min,      max,           default,    decimals, step
        ("BHP_psi",          "BHP (psi)",           150.0,    2000.0,        800.0,      2,  10.0),
        ("Area_ft2",         "Area (ft²)",          1.73e3,   3.53e5,        1.5e3,      2,  100.0),
        ("InjRate_ft3_day",  "Inj. Rate (ft³/day)", 1.0e6,    2.49e7,        7.0e6,      2,  1.0e5),
        ("POR",              "Porosity (-)",        0.02,     0.40,          0.20,       4,  0.005),
        ("Perm_mD",          "Permeability (mD)",   0.05,     500.0,         30.0,       3,  1.0),
        ("Thickness_ft",     "Thickness (ft)",      40.0,     500.0,         100.0,      2,  5.0),
        ("Depth_ft",         "Depth (ft)",          3500.0,   6500.0,        4000.0,     1,  50.0),
        ("S_org",            "S_org (-)",           0.10,     0.20,          0.15,       4,  0.005),
        ("S_orw",            "S_orw (-)",           0.20,     0.41,          0.25,       4,  0.005),
    ]

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.rows = {}  # key -> SliderRow

        # Debounce timer so we don't recompute on every microscopic slider tick
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(80)
        self._debounce.timeout.connect(self._do_predict)

        self._build_ui()

    # -- UI build --
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        # ----- Top: live result card -----
        result_card = QGroupBox("Live Predictions")
        rgrid = QGridLayout(result_card)
        rgrid.setHorizontalSpacing(20)
        rgrid.setVerticalSpacing(6)

        lbl_co2 = QLabel("CO₂ Storage (tons)")
        lbl_co2.setObjectName("liveLabel")
        lbl_oil = QLabel("Cumm. Oil Production (bbl)")
        lbl_oil.setObjectName("liveLabel")

        self.lbl_co2_val = QLabel("—")
        self.lbl_co2_val.setObjectName("liveValue")
        self.lbl_oil_val = QLabel("—")
        self.lbl_oil_val.setObjectName("liveValue")

        rgrid.addWidget(lbl_co2, 0, 0, alignment=Qt.AlignLeft)
        rgrid.addWidget(self.lbl_co2_val, 1, 0, alignment=Qt.AlignLeft)
        rgrid.addWidget(lbl_oil, 0, 1, alignment=Qt.AlignLeft)
        rgrid.addWidget(self.lbl_oil_val, 1, 1, alignment=Qt.AlignLeft)

        outer.addWidget(result_card)

        # ----- Middle: scrollable parameter form -----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        container.setObjectName("body")
        scroll.setWidget(container)
        outer.addWidget(scroll, stretch=1)

        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(8, 4, 8, 12)

        form_group = QGroupBox("Reservoir & Operational Inputs")
        grid = QGridLayout(form_group)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(12)

        # Column headers
        for col, name in enumerate(["Parameter", "Slider", "Value"]):
            h = QLabel(name)
            h.setObjectName("colHeader")
            grid.addWidget(h, 0, col)

        for r, (key, label, vmin, vmax, default, dec, step) in enumerate(
                self.PARAM_DEFAULTS, start=1):
            row = SliderRow(
                label_text=label,
                vmin=vmin,
                vmax=vmax,
                default=default,
                decimals=dec,
                step=step,
                on_change=self._on_value_changed,
            )
            self.rows[key] = row
            grid.addWidget(row.label, r, 0, alignment=Qt.AlignVCenter)
            grid.addWidget(row.slider, r, 1)
            grid.addWidget(row.box, r, 2)

        vbox.addWidget(form_group)

        # ----- Bottom: action buttons -----
        btn_row = QHBoxLayout()
        btn_row.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.reset_btn = QPushButton("Reset to defaults")
        self.reset_btn.clicked.connect(self._reset_defaults)
        self.predict_btn = QPushButton("Predict now")
        self.predict_btn.clicked.connect(self._do_predict)
        btn_row.addWidget(self.reset_btn)
        btn_row.addWidget(self.predict_btn)
        outer.addLayout(btn_row)

        # Try to load models eagerly so we can rebind slider ranges to the
        # training X_min / X_max from the JSON files. If they aren't there
        # yet, fall back to the manuscript defaults.
        self._try_bind_to_model_ranges()

        # Trigger an initial prediction
        QTimer.singleShot(100, self._do_predict)

    # -- behaviour --
    def _on_value_changed(self):
        self._debounce.start()

    def _do_predict(self):
        try:
            self.main_window.engine.ensure_loaded()
        except FileNotFoundError as e:
            self.lbl_co2_val.setText("model not loaded")
            self.lbl_oil_val.setText("model not loaded")
            self.main_window.status.showMessage(str(e), 6000)
            return
        except Exception as e:
            self.lbl_co2_val.setText("error")
            self.lbl_oil_val.setText("error")
            self.main_window.status.showMessage(f"Load error: {e}", 6000)
            return

        try:
            co2, oil = self.main_window.engine.predict(self.values())
            self.lbl_co2_val.setText(self._fmt(co2))
            self.lbl_oil_val.setText(self._fmt(oil))
            self.main_window.output_tab.set_values(co2, oil)
            self.main_window.output_tab.set_model_status(
                self.main_window.engine.co2_model_path,
                self.main_window.engine.oil_model_path,
            )
        except Exception as e:
            self.lbl_co2_val.setText("error")
            self.lbl_oil_val.setText("error")
            self.main_window.status.showMessage(f"Predict error: {e}", 6000)

    def _try_bind_to_model_ranges(self):
        """If the model JSONs are present, rebind each slider's [min, max]
        to the X_min/X_max stored in the model so the user can never give
        the network an out-of-range input."""
        try:
            self.main_window.engine.ensure_loaded()
        except Exception:
            return  # Models not yet present; keep manuscript defaults
        eng = self.main_window.engine
        # Use the CO2 model's X_min/X_max as the canonical envelope
        # (CO2 and Oil were trained on the same inputs).
        x_min = eng.co2_model["X_min"]
        x_max = eng.co2_model["X_max"]
        for i, (key, *_rest) in enumerate(self.PARAM_DEFAULTS):
            self.rows[key].set_range(float(x_min[i]), float(x_max[i]))
        self.main_window.status.showMessage("Slider ranges bound to model envelope", 4000)

    def _reset_defaults(self):
        for key, _label, _vmin, _vmax, default, *_ in self.PARAM_DEFAULTS:
            self.rows[key].set_value(default)
        self._do_predict()

    def values(self) -> dict:
        return {k: self.rows[k].value() for k, *_ in self.PARAM_DEFAULTS}

    @staticmethod
    def _fmt(x: float) -> str:
        if x == 0:
            return "0"
        if abs(x) >= 1e6:
            return f"{x:,.3e}"
        if abs(x) >= 1000:
            return f"{x:,.1f}"
        return f"{x:.4f}"


# =============================================================================
# Output / status tab
# =============================================================================
class OutputTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        self.card_pred = QGroupBox("Latest Predictions")
        grid = QGridLayout(self.card_pred)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(12)

        self.lbl_co2 = QLabel("CO₂ Storage (tons):")
        self.val_co2 = QLabel("—")
        self.lbl_oil = QLabel("Cumm. Oil Prod (bbl):")
        self.val_oil = QLabel("—")

        f_big = self.val_co2.font()
        f_big.setPointSize(22)
        f_big.setBold(True)
        self.val_co2.setFont(f_big)
        self.val_oil.setFont(f_big)

        grid.addWidget(self.lbl_co2, 0, 0)
        grid.addWidget(self.val_co2, 0, 1)
        grid.addWidget(self.lbl_oil, 1, 0)
        grid.addWidget(self.val_oil, 1, 1)

        self.card_status = QGroupBox("Model status")
        g2 = QGridLayout(self.card_status)
        self.lbl_co2_status = QLabel("CO₂ model: not loaded")
        self.lbl_oil_status = QLabel("Oil model: not loaded")
        g2.addWidget(self.lbl_co2_status, 0, 0)
        g2.addWidget(self.lbl_oil_status, 1, 0)

        layout.addWidget(self.card_pred)
        layout.addWidget(self.card_status)
        layout.addStretch(1)

    @staticmethod
    def _fmt(x: float) -> str:
        if abs(x) >= 1000:
            return f"{x:,.3f}".rstrip("0").rstrip(".")
        return f"{x:.6f}".rstrip("0").rstrip(".")

    def set_values(self, co2: float, oil: float):
        self.val_co2.setText(self._fmt(co2))
        self.val_oil.setText(self._fmt(oil))

    def set_model_status(self, co2_path: str = None, oil_path: str = None):
        self.lbl_co2_status.setText(
            f"CO₂ model: {'loaded (' + os.path.basename(co2_path) + ')' if co2_path else 'not loaded'}"
        )
        self.lbl_oil_status.setText(
            f"Oil model: {'loaded (' + os.path.basename(oil_path) + ')' if oil_path else 'not loaded'}"
        )


# =============================================================================
# Tornado bar widget (custom-painted, no matplotlib)
# =============================================================================
class TornadoWidget(QWidget):
    """Horizontal bar chart of variance contribution per input parameter.

    Receives a list of (label, value) tuples plus a `units` string and draws
    one horizontal bar per row with a teal fill. The bars are normalized to
    the largest value so the longest bar fills the available width.
    """
    BG = QColor("#13254d")
    BAR = QColor("#2fb4c8")
    BAR_BG = QColor("#0f1d3c")
    TEXT = QColor("#e6eefc")
    AXIS = QColor("#1f3b73")

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        self._rows = []          # list of (label, value)
        self._units = ""
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)

    def set_data(self, rows, units: str = ""):
        # Sort descending by value so the largest contributor is on top
        self._rows = sorted(rows, key=lambda r: r[1], reverse=True)
        self._units = units
        # Auto-resize: ~28 px per row + 50 px header
        self.setMinimumHeight(max(220, 50 + 28 * len(self._rows)))
        self.update()

    def sizeHint(self):
        return QSize(420, max(220, 50 + 28 * max(1, len(self._rows))))

    def paintEvent(self, _evt):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        p.fillRect(rect, self.BG)

        # Title
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        p.setFont(title_font)
        p.setPen(self.TEXT)
        p.drawText(12, 22, self._title)

        if not self._rows:
            p.setPen(QColor("#b8c3e0"))
            small = QFont(); small.setPointSize(10); p.setFont(small)
            p.drawText(12, 52, "Run UQ to populate this chart.")
            return

        max_val = max((v for _l, v in self._rows), default=1.0) or 1.0
        label_w = 130   # left column width for parameter names
        margin_l = 12
        margin_r = 14
        bar_x = margin_l + label_w
        bar_w = max(40, rect.width() - bar_x - margin_r - 90)  # leave room for value text
        row_h = 26
        y = 42

        label_font = QFont(); label_font.setPointSize(10); p.setFont(label_font)
        for label, value in self._rows:
            # Label
            p.setPen(self.TEXT)
            p.drawText(margin_l, y + row_h - 8, label)

            # Bar background
            p.fillRect(bar_x, y + 4, bar_w, row_h - 10, self.BAR_BG)
            # Bar fill
            frac = (value / max_val) if max_val > 0 else 0
            fill_w = int(round(bar_w * frac))
            if fill_w > 0:
                p.fillRect(bar_x, y + 4, fill_w, row_h - 10, self.BAR)

            # Value text on the right
            p.setPen(self.TEXT)
            txt = self._fmt(value) + (f" {self._units}" if self._units else "")
            p.drawText(bar_x + bar_w + 6, y + row_h - 8, txt)

            y += row_h
        p.end()

    @staticmethod
    def _fmt(x: float) -> str:
        if x == 0:
            return "0"
        if abs(x) >= 1e6 or (abs(x) > 0 and abs(x) < 0.01):
            return f"{x:.2e}"
        if abs(x) >= 1000:
            return f"{x:,.1f}"
        return f"{x:.3f}"


# =============================================================================
# Uncertainty propagation tab (Monte Carlo / LHS over per-parameter CVs)
# =============================================================================
class UncertaintyTab(QWidget):
    """Monte-Carlo uncertainty propagation through the trained ANN.

    For each parameter, the user specifies a coefficient of variation (CV %).
    On Run, the tab draws N samples from a truncated-normal (or uniform)
    distribution around the current slider value, runs the vectorized ANN
    forward pass, and reports for both targets:
        mean, std, P10, P50, P90, 95% empirical CI lower/upper.
    A one-at-a-time main-effects tornado plot shows which inputs contribute
    most to the spread.
    """
    PARAM_KEYS = [k for k, *_ in InputTab.PARAM_DEFAULTS]
    PARAM_LABELS = {k: lbl for k, lbl, *_ in InputTab.PARAM_DEFAULTS}

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.cv_boxes = {}      # key -> QDoubleSpinBox (% CV)
        self._build_ui()

    # -- UI --
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        # Intro
        intro = QLabel(
            "Set a per-parameter coefficient of variation (CV, % of slider value), "
            "then click Run. Samples are drawn around the current Inputs (Live) slider "
            "values and clipped to the model training envelope."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        # ----- Per-parameter CV grid -----
        cv_group = QGroupBox("Per-parameter uncertainty (CV %)")
        cv_grid = QGridLayout(cv_group)
        cv_grid.setHorizontalSpacing(20)
        cv_grid.setVerticalSpacing(8)

        h0 = QLabel("Parameter"); h0.setObjectName("colHeader")
        h1 = QLabel("CV (%)");    h1.setObjectName("colHeader")
        cv_grid.addWidget(h0, 0, 0)
        cv_grid.addWidget(h1, 0, 1)
        cv_grid.addWidget(QLabel(""), 0, 2)  # spacer

        h0b = QLabel("Parameter"); h0b.setObjectName("colHeader")
        h1b = QLabel("CV (%)");    h1b.setObjectName("colHeader")
        cv_grid.addWidget(h0b, 0, 3)
        cv_grid.addWidget(h1b, 0, 4)

        # Two columns of 5 + 4 = 9 parameters
        half = (len(self.PARAM_KEYS) + 1) // 2
        for i, key in enumerate(self.PARAM_KEYS):
            col = 0 if i < half else 3
            row = 1 + (i if i < half else i - half)
            lbl = QLabel(self.PARAM_LABELS[key])
            box = QDoubleSpinBox()
            box.setRange(0.0, 100.0)
            box.setDecimals(2)
            box.setSingleStep(1.0)
            box.setValue(0.0)
            box.setSuffix(" %")
            box.setObjectName("cvBox")
            box.setButtonSymbols(QDoubleSpinBox.NoButtons)
            cv_grid.addWidget(lbl, row, col)
            cv_grid.addWidget(box, row, col + 1)
            self.cv_boxes[key] = box

        outer.addWidget(cv_group)

        # ----- Controls row -----
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(14)

        ctrl_row.addWidget(QLabel("Distribution:"))
        self.dist_combo = QComboBox()
        self.dist_combo.addItem("Truncated normal", "tnormal")
        self.dist_combo.addItem("Uniform (Table-1 envelope)", "uniform")
        ctrl_row.addWidget(self.dist_combo)

        ctrl_row.addItem(QSpacerItem(20, 1, QSizePolicy.Fixed, QSizePolicy.Minimum))
        ctrl_row.addWidget(QLabel("N samples:"))
        self.n_box = QSpinBox()
        self.n_box.setRange(100, 200000)
        self.n_box.setSingleStep(1000)
        self.n_box.setValue(10000)
        self.n_box.setButtonSymbols(QSpinBox.NoButtons)
        self.n_box.setMinimumWidth(110)
        ctrl_row.addWidget(self.n_box)

        ctrl_row.addItem(QSpacerItem(20, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.zero_btn = QPushButton("Zero all CVs")
        self.zero_btn.clicked.connect(self._zero_all)
        ctrl_row.addWidget(self.zero_btn)

        self.run_btn = QPushButton("Run Uncertainty Propagation")
        self.run_btn.clicked.connect(self._on_run)
        ctrl_row.addWidget(self.run_btn)

        outer.addLayout(ctrl_row)

        # ----- Results group -----
        res_group = QGroupBox("UQ Results")
        rgrid = QGridLayout(res_group)
        rgrid.setHorizontalSpacing(28)
        rgrid.setVerticalSpacing(0)

        STAT_ROWS = [
            ("Mean",        "mean"),
            ("Std. dev.",   "std"),
            ("P10",         "p10"),
            ("P50 (median)", "p50"),
            ("P90",         "p90"),
            ("95% CI lower", "ci_lo"),
            ("95% CI upper", "ci_hi"),
            ("Clip fraction", "clip"),
        ]
        # Headers
        hdr_param = QLabel("Statistic"); hdr_param.setObjectName("colHeader")
        hdr_co2 = QLabel("CO₂ Storage (tons)"); hdr_co2.setObjectName("colHeader")
        hdr_oil = QLabel("Cumm. Oil (bbl)");    hdr_oil.setObjectName("colHeader")
        rgrid.addWidget(hdr_param, 0, 0)
        rgrid.addWidget(hdr_co2,   0, 1)
        rgrid.addWidget(hdr_oil,   0, 2)

        self.stat_labels = {}  # key -> (co2_label, oil_label)
        for i, (label_text, key) in enumerate(STAT_ROWS, start=1):
            stat_lbl = QLabel(label_text)
            stat_lbl.setObjectName("uqStatLabel")
            rgrid.addWidget(stat_lbl, i, 0)
            lc = QLabel("—"); lc.setMinimumWidth(180); lc.setObjectName("uqStatValue")
            lo = QLabel("—"); lo.setMinimumWidth(180); lo.setObjectName("uqStatValue")
            rgrid.addWidget(lc, i, 1)
            rgrid.addWidget(lo, i, 2)
            self.stat_labels[key] = (lc, lo)

        outer.addWidget(res_group)

        # ----- Tornado plots side by side -----
        tornado_group = QGroupBox("Variance contribution (one-at-a-time main effects)")
        trow = QHBoxLayout(tornado_group)
        trow.setSpacing(14)
        self.tornado_co2 = TornadoWidget("CO₂ storage  —  std contribution")
        self.tornado_oil = TornadoWidget("Cumulative oil  —  std contribution")
        trow.addWidget(self.tornado_co2, 1)
        trow.addWidget(self.tornado_oil, 1)
        outer.addWidget(tornado_group, stretch=1)

    # -- behaviour --
    def _zero_all(self):
        for box in self.cv_boxes.values():
            box.setValue(0.0)

    def _on_run(self):
        try:
            self.main_window.engine.ensure_loaded()
        except Exception as e:
            QMessageBox.critical(self, "Models not loaded",
                                 f"Cannot run UQ until models are loaded.\n{e}")
            return

        # Read current slider values from the Input tab as the central values
        mu_dict = self.main_window.input_tab.values()
        mu = np.array([mu_dict[k] for k in self.PARAM_KEYS], dtype=float)

        # Per-parameter sigma derived from CV (% of mean)
        cv = np.array([self.cv_boxes[k].value() / 100.0 for k in self.PARAM_KEYS], dtype=float)
        sigma = np.abs(mu) * cv

        if np.all(sigma == 0):
            QMessageBox.information(
                self, "All CVs are zero",
                "Set a non-zero CV on at least one parameter to propagate uncertainty.",
            )
            return

        n = int(self.n_box.value())
        dist = self.dist_combo.currentData()
        x_min, x_max = self.main_window.engine.envelope()

        rng = np.random.default_rng(0)  # deterministic; change to None for random seed
        try:
            X, clip_frac = self._sample(mu, sigma, x_min, x_max, n, dist, rng)
        except Exception as e:
            QMessageBox.critical(self, "Sampling error", str(e))
            return

        # Vectorized forward pass
        try:
            co2, oil = self.main_window.engine.predict_batch(X)
        except Exception as e:
            QMessageBox.critical(self, "Prediction error", str(e))
            return

        # Update result labels
        self._set_stats("co2", co2, clip_frac)
        self._set_stats("oil", oil, clip_frac)

        # One-at-a-time tornado: for each input i with sigma_i > 0, sample only i
        # (others held at mu) and report the std of CO2 and Oil predictions.
        co2_contrib = []
        oil_contrib = []
        for i, key in enumerate(self.PARAM_KEYS):
            if sigma[i] == 0:
                continue
            sigma_i = np.zeros_like(sigma)
            sigma_i[i] = sigma[i]
            Xi, _ = self._sample(mu, sigma_i, x_min, x_max, max(2000, n // 5), dist, rng)
            ci, oi = self.main_window.engine.predict_batch(Xi)
            co2_contrib.append((self.PARAM_LABELS[key], float(np.std(ci))))
            oil_contrib.append((self.PARAM_LABELS[key], float(np.std(oi))))

        self.tornado_co2.set_data(co2_contrib, units="tons")
        self.tornado_oil.set_data(oil_contrib, units="bbl")

        self.main_window.status.showMessage(
            f"UQ complete: {n} samples, distribution={dist}", 6000
        )

    # -- sampling --
    @staticmethod
    def _sample(mu, sigma, x_min, x_max, n, dist, rng):
        """Draw N LHS samples in 9-D and clip to the training envelope.

        Returns (X of shape (N, D), clip_frac of shape (D,)).
        """
        D = len(mu)
        X = np.zeros((n, D))
        clip_frac = np.zeros(D)
        for d in range(D):
            if sigma[d] == 0:
                X[:, d] = mu[d]
                continue

            # LHS in [0, 1] with random offset within each stratum, then shuffled
            u = (np.arange(n) + rng.uniform(0.0, 1.0, size=n)) / n
            rng.shuffle(u)

            if dist == "tnormal":
                # Map [0,1] LHS through inverse standard normal CDF, then shift/scale
                u_clipped = np.clip(u, 1e-9, 1 - 1e-9)
                z = norm.ppf(u_clipped)
                vals = mu[d] + sigma[d] * z
            elif dist == "uniform":
                # Uniform across [mu - sqrt(3)*sigma, mu + sqrt(3)*sigma]
                # so the std matches the requested sigma exactly.
                half_width = np.sqrt(3.0) * sigma[d]
                vals = (mu[d] - half_width) + 2.0 * half_width * u
            else:
                raise ValueError(f"Unknown distribution: {dist}")

            n_out = int(np.sum((vals < x_min[d]) | (vals > x_max[d])))
            clip_frac[d] = n_out / n
            X[:, d] = np.clip(vals, x_min[d], x_max[d])
        return X, clip_frac

    # -- formatting --
    def _set_stats(self, which, arr, clip_frac):
        """which: 'co2' or 'oil'."""
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        p10, p50, p90 = (float(x) for x in np.percentile(arr, [10, 50, 90]))
        ci_lo, ci_hi = (float(x) for x in np.percentile(arr, [2.5, 97.5]))
        max_clip = float(np.max(clip_frac))

        vals = {
            "mean": mean, "std": std, "p10": p10, "p50": p50, "p90": p90,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "clip": max_clip,
        }

        for key, (lc, lo) in self.stat_labels.items():
            target_label = lc if which == "co2" else lo
            v = vals[key]
            if key == "clip":
                target_label.setText(f"{v*100:.2f} %  (max over inputs)")
                if v > 0.05:
                    target_label.setStyleSheet("color: #ffb86c; font-weight: 700; font-size: 20px;")
                else:
                    target_label.setStyleSheet("color: #2fb4c8; font-weight: 700; font-size: 20px;")
            else:
                target_label.setText(self._fmt(v))

    @staticmethod
    def _fmt(x: float) -> str:
        if x == 0:
            return "0"
        if abs(x) >= 1e6:
            return f"{x:,.3e}"
        if abs(x) >= 1000:
            return f"{x:,.1f}"
        return f"{x:.4f}"


# =============================================================================
# Main window
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCS-EOROptTool v1 — Live Slider + UQ")
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(load_stylesheet())

        self.engine = PredictionEngine()

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.input_tab = InputTab(self, self)
        self.uq_tab = UncertaintyTab(self)
        self.output_tab = OutputTab(self)

        self.tabs.addTab(self.input_tab, "Inputs (Live)")
        self.tabs.addTab(self.uq_tab, "Uncertainty")
        self.tabs.addTab(self.output_tab, "Status")

        self.status.showMessage("Move sliders to update predictions; switch to Uncertainty for Monte Carlo")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CCS-EOROptTool v1")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()