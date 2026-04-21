# CCS-EOROptTool

**CCS-EOROptTool** is a user-friendly graphical user interface (GUI) application is designed as a surrogate to provide fast and intelligent predictions of CO₂ storage capacity and cumulative oil production in unconventional reservoirs using a trained artificial neural network (ANN).

This tool supports decision-making in Carbon Capture and Storage with Enhanced Oil Recovery (CCS-EOR) by enabling quick assessments based on key reservoir and operational parameters. The model is both explainable and explicit, offering transparency and interpretability that aligns with engineering best practices.

---

## 🌍 Project Purpose

This work aims to develop an explainable, transparent, and data-driven modeling tool to:

* Predict CO₂ storage mass and cumulative oil production.
* Enable quick sensitivity analysis and scenario planning.
* Assist reservoir engineers and CCS planners with smart, real-time decision support.

---

## 🧠 Features

* Built-in **trained ANN model** with high predictive accuracy (R² > 0.96).
* Simple GUI interface with user input fields for key reservoir parameters.
* Real-time prediction of CO₂ storage and oil production with uncertainty bands.
* Light-weight and fully implemented in **Python**, with no need for advanced ML expertise.

---

## 🖥️ Requirements

Make sure you have the following installed:

* Python 3.7 or above
* Required Python libraries (install with pip if missing):

```bash
pip install numpy pandas matplotlib tkinter joblib PyQt5
```

> **Note**: `tkinter` comes pre-installed with most Python distributions. If you're using a minimal install (like some Linux distros), you may need to install it separately.

---

## 📁 Files in This Repository

| File                        | Description                                         |
| --------------------------- | --------------------------------------------------- |
| `CCS-EOROptTool GUI_v1.py`  | The main GUI application script.                    |
| `trainedNN_FullCO2.json`    | Pre-trained ANN CO2 weights file (used by the GUI). |
| `trainedNN_FullOil.json`    | Pre-trained ANN Oil weights file (used by the GUI). |
| `README.md`                 | This readme file.                                   |

---

## 🚀 How to Run

1. **Clone or download** the repository to your local machine.

2. Ensure the `trained_model_weights.pkl` file is in the same directory as the `CCS-EOROptTool GUI.py`.

3. Open a terminal or command prompt and navigate to the project directory.

4. Run the Python script:

```bash
python "CCS-EOROptTool GUI_v1.py"
```

The GUI window will open, allowing you to input key reservoir parameters such as:

* Depth
* Porosity
* Permeability
* Thickness
* Bottom-hole pressure
* CO₂ injection rate
* Sorw (residual oil saturation)
* ...and others

Once inputs are entered, the tool will predict:

* **CO₂ storage mass**
* **Cumulative oil production**

Predictions are displayed along with uncertainty intervals, offering a robust and explainable decision-support tool for CCS-EOR operations.

---

## 📬 Feedback & Contributions

We welcome contributions, suggestions, and feedback. Please feel free to open issues or submit pull requests.

---

## 📄 License

This project is licensed under [MIT License](LICENSE).

---

