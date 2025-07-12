# 🚀 Laptop Price Predictor Project

Welcome! This is a **Machine Learning project** to predict laptop prices based on various features.  
It’s designed to run fully **inside a container** using **Miniforge, Conda, and Mamba**, so you don’t have to install Python or libraries locally.  
All work is done in **VS Code Dev Containers** — including running Jupyter notebooks and the Flask app!

---

## 📦 Project Highlights

✅ Machine Learning pipeline for predicting laptop prices  
✅ Data preprocessing and EDA in **Jupyter notebooks**  
✅ Flask web app with a simple prediction UI  
✅ Fully containerized with **Miniforge + Mamba**  
✅ Ready to run in **VS Code Dev Container**

---

## 📂 Project Structure

```
laptop_price_predictor_project/
├── Makefile                  # Useful commands for build/test (WIP)
├── pyproject.toml            # Project metadata & dependencies
├── README.md                 # Project documentation (you’re reading it!)
├── .devcontainer.json        # Devcontainer config for VS Code
├── app/                      # Flask app for web deployment
│   ├── app.py                # Main Flask application
│   ├── static/style.css      # Stylesheet
│   └── templates/index.html  # HTML template for the web UI
├── data/                     # Raw dataset(s)
│   └── laptop_data.csv
├── notebooks/                # Experiments & preprocessing notebooks
│   ├── experiment_01.ipynb
│   ├── FC212012-Sajani/
│   ├── FC212017-Dasun/
│   ├── FC212030_Mudhitha/
│   ├── FC212034_Kishgintharaam/
│   └── FC212036_Dileesha/
├── src/                      # Source code for training, validation, utils
│   ├── __init__.py
│   ├── test.py
│   ├── train.py
│   ├── utils.py
│   ├── validation.py
│   ├── FC212017-Dasun/       # Individual contribution folder
│   └── FC212034_Kishgintharaam/ # Individual contribution folder
└── tests/                    # Unit tests
    ├── __init__.py
    └── test.py

```

---

## ⚙️ How To Use

### ✅ Prerequisites

- [Visual Studio Code](https://code.visualstudio.com/)
- [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

---

### 🚀 Quick Start

1️⃣ **Clone the repository**

```bash
git clone https://github.com/ddasunsandeepa/laptop_price_predictor_project.git
cd laptop_price_predictor_project
```

2️⃣ **Open in VS Code**

- Open the project folder in VS Code.
- When prompted, **"Reopen in Container"** — click it.
- VS Code will build the container image using `.devcontainer.json` and launch the dev environment.

3️⃣ **Install dependencies**

Inside the container terminal, run:

```bash
# Example: If you have an environment.yml
mamba env create -f environment.yml

# Or, install packages manually
mamba install numpy pandas scikit-learn flask jupyter
```

_(Adjust the packages to what you actually need.)_

4️⃣ **Launch Jupyter Notebook**

```bash
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

Copy the URL output and open it in your local browser.

5️⃣ **Run the Flask app**

```bash
cd app
python app.py
```

Visit [http://localhost:5000](http://localhost:5000) to use the predictor!

---

## 🧩 How It Works

- All dependencies are installed with **Mamba**, a faster drop-in replacement for Conda.
- Notebooks for data cleaning, EDA, and model training are in `notebooks/`.
- Training, validation, and utility code lives in `src/`.
- The Flask app in `app/` loads trained `.pkl` models to make predictions.
- Data files are in `data/`.

---

## ⚡ Tips

✔️ **No Python setup needed** — everything runs inside the container!  
✔️ You can develop, train, validate, test, and deploy all from the same containerized environment.  
✔️ Add a `requirements.txt` or `environment.yml` to easily share your exact dependencies.

---

## 🧪 Development Workflow

- **Explore & preprocess data:** Use notebooks in `notebooks/`
- **Train & validate models:** Run scripts in `src/`
- **Test your code:** Use `tests/` folder
- **Deploy & test Flask app:** Use `app/`

---

## ✅ To Do

- Add a `requirements.txt` or `environment.yml` file for consistent environments.
- Add automated unit tests in `tests/`.
- Add CI/CD (GitHub Actions).
- Deploy the Flask app to Heroku, Render, or similar.

---

## 👨‍💻 Authors

- **Dasun Sandeepa** — Dsk.sandeep.987@gmail.com
- **Kishgintharaam Sathannthan** — kishgi1234@gmail.com

---

## 📜 License

This project is open-source. Add your license here (e.g., MIT).

---

**Happy Predicting! 🚀**
