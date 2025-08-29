"""
CO2 Storage and Cumulative Oil Production GUI
---------------------------------------------
PyQt5 UI that auto-loads MATLAB-exported ANN models (JSON) the first time
you click Predict. It looks for:
  - TrainedNN_FullCO2.json  -> predicts CO2 Storage (ton)
  - TrainedNN_FullOil.json  -> predicts Cumm. Oil Prod (bbl)
"""

import sys
import os
import json
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QDoubleSpinBox,
    QPushButton,
    QGroupBox,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QScrollArea,
    QFrame,
    QMessageBox,
    QStatusBar,
)


# --------------------------- Styling ---------------------------------
def load_stylesheet() -> str:
    """Rich blue theme with large, readable fonts."""
    blue_900 = "#0f1d3c"  # page background
    blue_800 = "#13254d"  # card background
    blue_600 = "#1f3b73"  # headers and accents
    teal_accent = "#2fb4c8"  # buttons and highlights
    text_primary = "#e6eefc"
    text_muted = "#b8c3e0"

    return f"""
        * {{
            font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            font-size: 18px;
        }}
        QMainWindow {{
            background: {blue_900};
        }}
        QAbstractScrollArea {{
            background: {blue_900};
            border: 0px;
        }}
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

        QLabel {{ color: {text_primary}; font-size: 22px; }}
        QLabel#colHeader {{
            color: {text_primary};     /* white & bold */
            font-size: 20px;
            font-weight: 800;
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
            padding: 10px 14px;
            min-height: 52px;
            font-size: 20px;
        }}
        QDoubleSpinBox:focus {{ border: 1px solid {teal_accent}; }}

        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {teal_accent}, stop:1 #1c8aa0);
            color: #081222;
            font-weight: 700;
            border: none;
            border-radius: 12px;
            padding: 14px 28px;
            font-size: 18px;
        }}
        QPushButton:hover {{ filter: brightness(1.08); }}
        QPushButton:pressed {{ filter: brightness(0.95); }}
    """


# ------------------------ Prediction Engine --------------------------
class PredictionEngine:
    """Auto-loads models from the script directory and runs predictions.
    Supports MATLAB ANN exported as JSON and (optionally) joblib .pkl.
    """
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
        self.co2_kind = None  # 'json' or 'pkl'
        self.oil_kind = None

        self.co2_model_path = None
        self.oil_model_path = None

    # -- Public API --
    def ensure_loaded(self):
        if self.co2_model is None:
            self._autoload("co2")
        if self.oil_model is None:
            self._autoload("oil")

    def predict(self, values: dict) -> tuple:
        self.ensure_loaded()
        X = self._row(values)

        co2_norm = self._predict_one(self.co2_model, self.co2_kind, X, values, which="co2")
        oil_norm = self._predict_one(self.oil_model, self.oil_kind, X, values, which="oil")

        # Use model-bound min/max values
        co2_real = self._denormalize(co2_norm, self.co2_model["co2_min"], self.co2_model["co2_max"])
        oil_real = self._denormalize(oil_norm, self.oil_model["oil_min"], self.oil_model["oil_max"])

        # Clamp negative predictions to zero
        co2_real = max(0.0, co2_real)
        oil_real = max(0.0, oil_real)

        return round(float(co2_real), 6), round(float(oil_real), 6)

    # -- Internals --
    def _row(self, values: dict):
        return np.array([[values[k] for k in self.FEATURE_ORDER]], dtype=float)

    def _autoload(self, which: str):
        path = self._find_model_path(which)
        if not path:
            want = "TrainedNN_FullCO2.json" if which.lower() == "co2" else "TrainedNN_FullOil.json"
            raise FileNotFoundError(
                f"Could not find a {which.upper()} model in {self.model_dir}. Expected '{want}'."
            )
        self._load_model(which, path)

    def _find_model_path(self, which: str):
        which_lower = which.lower()
        exact_map = {
            "co2": "TrainedNN_FullCO2.json",
            "oil": "TrainedNN_FullOil.json",
        }
        exact = os.path.join(self.model_dir, exact_map.get(which_lower, ""))
        if exact and os.path.isfile(exact):
            return exact

        # Fallbacks / variants (kept for flexibility)
        preferred = [
            f"TrainedNN_Full{which.upper()}.json",
            f"{which_lower}_model.json",
            f"{which_lower}.json",
            f"{which_lower}_model.pkl",
            f"{which_lower}.pkl",
        ]
        for name in preferred:
            p = os.path.join(self.model_dir, name)
            if os.path.isfile(p):
                return p

        # Last resort: scan for any file containing the token
        for ext in (".json", ".pkl"):
            for fname in os.listdir(self.model_dir):
                if fname.lower().endswith(ext) and which_lower in fname.lower():
                    return os.path.join(self.model_dir, fname)
        return None

    def _load_model(self, which: str, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            model, kind = self._load_matlab_json(path), "json"
        elif ext == ".pkl":
            try:
                import joblib  # optional dependency
            except Exception as e:
                raise RuntimeError("joblib is required for .pkl models. Run: pip install joblib") from e
            model, kind = joblib.load(path), "pkl"
        else:
            raise ValueError("Unsupported model file. Use .json or .pkl.")

        if which == "co2":
            self.co2_model, self.co2_kind, self.co2_model_path = model, kind, path
        else:
            self.oil_model, self.oil_kind, self.oil_model_path = model, kind, path

    @staticmethod
    def _tansig(x):
        return np.tanh(x)

    def _load_matlab_json(self, path: str):
        with open(path, "r") as f:
            nn = json.load(f)

        iw = np.asarray(nn["IW"], dtype=float)
        lw = np.asarray(nn["LW"], dtype=float).reshape(1, -1)
        b = nn["b"]
        b1 = np.asarray(b[0], dtype=float).reshape(-1)
        b2 = float(np.asarray(b[1], dtype=float))
        x_min = np.asarray(nn["X_min"], dtype=float).reshape(-1)
        x_max = np.asarray(nn["X_max"], dtype=float).reshape(-1)
        hidden = nn.get("hiddenActivation", "tansig").lower()
        out = nn.get("outputActivation", "purelin").lower()

        # Instead of assigning to self
        co2_min = float(nn.get("CO2_min", 0))
        co2_max = float(nn.get("CO2_max", 1))
        oil_min = float(nn.get("Oil_min", 0))
        oil_max = float(nn.get("Oil_max", 1))

        if iw.shape[1] != len(self.FEATURE_ORDER):
            raise ValueError(f"Model expects {iw.shape[1]} features but GUI has {len(self.FEATURE_ORDER)}.")
        if lw.shape[1] != iw.shape[0] or b1.shape[0] != iw.shape[0]:
            raise ValueError("Dimension mismatch between IW, LW, and biases.")

        return {
            "IW": iw, "LW": lw, "b1": b1, "b2": b2,
            "X_min": x_min, "X_max": x_max,
            "hidden": hidden, "out": out,
            "co2_min": co2_min,
            "co2_max": co2_max,
            "oil_min": oil_min,
            "oil_max": oil_max
        }

    def _denormalize(self, norm_val: float, vmin: float, vmax: float) -> float:
            return norm_val * (vmax - vmin) + vmin

    def _predict_json(self, model: dict, X: np.ndarray) -> float:
        x_vec = X.reshape(-1)
        denom = model["X_max"] - model["X_min"]
        denom = np.where(denom == 0, 1.0, denom)  # avoid divide-by-zero
        x_norm = (x_vec - model["X_min"]) / denom

        h = np.dot(model["IW"], x_norm) + model["b1"]
        if model["hidden"] == "tansig":
            h = self._tansig(h)
        else:
            raise ValueError(f"Unsupported hidden activation: {model['hidden']}")

        y = np.dot(model["LW"], h) + model["b2"]  # 'purelin'
        return float(np.ravel(y)[0])

    def _predict_one(self, model, kind: str, X: np.ndarray, values: dict, which: str):
        if model is None:
            # Fallback: keep UI usable even if model missing
            if which == "co2":
                return 0.001 * values["Area_ft2"] * values["POR"] * (values["Thickness_ft"] + 1)
            return 5.0 * values["Perm_mD"] * (1 - values["S_org"]) * (1 - values["S_orw"]) + 0.1 * values["BHP_psi"]
        if kind == "json":
            return self._predict_json(model, X)
        # kind == 'pkl'
        return float(np.ravel(model.predict(X))[0])


# ------------------------------ UI -----------------------------------
class InputTab(QWidget):
    """Input form with parameters and a Predict button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _add_numeric(
        self, grid: QGridLayout, row: int, label: str,
        minimum: float, maximum: float, step: float, decimals: int,
        default: float
    ) -> QDoubleSpinBox:
        lab = QLabel(label)
        lab.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(11)
        box.setSingleStep(step)
        box.setValue(default)
        box.setButtonSymbols(QDoubleSpinBox.NoButtons)
        box.setMinimumHeight(44)

        grid.addWidget(lab, row, 0, alignment=Qt.AlignVCenter | Qt.AlignLeft)
        grid.addWidget(box, row, 1)
        return box

    def _build_ui(self):
        outer = QVBoxLayout(self)

        # Scrollable form content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        container.setObjectName("body")
        scroll.setWidget(container)
        outer.addWidget(scroll)

        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(16, 4, 16, 12)
        vbox.setSpacing(6)

        form_group = QGroupBox()
        grid = QGridLayout(form_group)
        grid.setContentsMargins(16, 10, 16, 16)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(14)

        # Column headers
        hdr_param = QLabel("Parameter (unit)")
        hdr_param.setObjectName("colHeader")
        hdr_value = QLabel("Value")
        hdr_value.setObjectName("colHeader")
        grid.addWidget(hdr_param, 0, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(hdr_value, 0, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)

        row = 1
        self.bhp     = self._add_numeric(grid, row, "BHP (psi)",           0, 20000, 10,   4, 0.0);      row += 1
        self.area    = self._add_numeric(grid, row, "Area (ft²)",          0, 1e8,   10,   4, 0.0);      row += 1
        self.injrate = self._add_numeric(grid, row, "InjRate (ft³/day)",   0, 1e8,   10,   4, 0.0);      row += 1
        self.por     = self._add_numeric(grid, row, "POR",                 0.0, 1.0, 0.01, 4, 0.0);      row += 1
        self.perm    = self._add_numeric(grid, row, "Perm (mD)",           0.0, 10000.0, 0.5, 4, 0.0);   row += 1
        self.thick   = self._add_numeric(grid, row, "Thickness (ft)",      0.0, 1000.0,  0.5, 4, 0.0);   row += 1
        self.depth   = self._add_numeric(grid, row, "Depth (ft)",          0.0, 30000.0, 1.0, 4, 0.0);   row += 1
        self.sorg    = self._add_numeric(grid, row, "S_org",               0.0, 1.0, 0.01, 4, 0.0);      row += 1
        self.sorw    = self._add_numeric(grid, row, "S_orw",               0.0, 1.0, 0.01, 4, 0.0);      row += 1

        vbox.addWidget(form_group)

        # Buttons row OUTSIDE the scroll area so it's always visible
        buttons_row = QHBoxLayout()
        buttons_row.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear_clicked)

        self.predict_btn = QPushButton("Predict")
        self.predict_btn.setToolTip("Run the model to predict CO₂ Storage and Cumulative Oil Production")
        self.predict_btn.clicked.connect(self._on_predict_clicked)

        buttons_row.addWidget(self.clear_btn)
        buttons_row.addWidget(self.predict_btn)

        outer.addLayout(buttons_row)   # <<— add to OUTER, not inside the scroll

    def values(self) -> dict:
        return {
            "BHP_psi": self.bhp.value(),
            "Area_ft2": self.area.value(),
            "InjRate_ft3_day": self.injrate.value(),
            "POR": self.por.value(),
            "Perm_mD": self.perm.value(),
            "Thickness_ft": self.thick.value(),
            "Depth_ft": self.depth.value(),
            "S_org": self.sorg.value(),
            "S_orw": self.sorw.value(),
        }

    def _on_predict_clicked(self):
        parent = self.parent()
        while parent and not isinstance(parent, MainWindow):
            parent = parent.parent()
        if isinstance(parent, MainWindow):
            try:
                parent.engine.ensure_loaded()  # auto-load JSON models
            except Exception as e:
                QMessageBox.critical(self, "Model load error", str(e))
                return
            parent.run_prediction(self.values())

    def _on_clear_clicked(self):
        for box in [self.bhp, self.area, self.injrate, self.por, self.perm,
                    self.thick, self.depth, self.sorg, self.sowv]:
            box.setValue(box.minimum())


class OutputTab(QWidget):
    """Shows predictions and model status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.card_pred = QGroupBox("Predictions")
        grid = QGridLayout(self.card_pred)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        self.lbl_co2 = QLabel("CO₂ Storage (tons) :")
        self.val_co2 = QLabel("—")
        self.lbl_oil = QLabel("Cumm. Oil Prod (bbl) :")
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

    def _fmt(self, x: float) -> str:
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCS-EOROptTool for Unconventional Reservoirs")
        self.setMinimumSize(920, 540)
        self.setStyleSheet(load_stylesheet())

        self.engine = PredictionEngine()

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        self.input_tab = InputTab(self)
        self.output_tab = OutputTab(self)

        tabs.addTab(self.input_tab, "Input")
        tabs.addTab(self.output_tab, "Output")

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Models will auto-load on first Predict")

    def run_prediction(self, values: dict):
        try:
            co2, oil = self.engine.predict(values)
            self.output_tab.set_values(co2, oil)
            self.output_tab.set_model_status(self.engine.co2_model_path, self.engine.oil_model_path)
            self.centralWidget().setCurrentIndex(1)
            self.status.showMessage("Prediction complete", 5000)
        except Exception as e:
            from traceback import format_exc
            QMessageBox.critical(self, "Prediction error", format_exc())


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CO₂-EORTool")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
