# 🚀 Laptop Price Predictor Project

This is a **Machine Learning project** that predicts laptop prices based on various features..  
The project is fully **containerized** using **Miniforge, Conda, and Mamba**, so you don’t need to install Python or libraries locally.  
It runs seamlessly in **VS Code Dev Containers**, including **Jupyter notebooks** and the **Flask web app**.



## 📦 Features

- Predict laptop prices using a trained ML pipeline  
- Data preprocessing and EDA in **Jupyter notebooks**  
- User-friendly **Flask web app** for price prediction  
- Fully containerized with **Miniforge + Conda**  
- Ready to run in **VS Code Dev Containers**  



## 📂 Project Structure

```

laptop_price_predictor_project/
├── .git/
├── .gitignore
├── .devcontainer.json
├── Makefile
├── Miniforge3-Linux-x86_64.sh
├── pyproject.toml
├── README.md
│
├── app/
│   ├── app.py
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
│
├── data/
│   └── laptop_data.csv
│
├── notebooks/
│   ├── FC212009_Darsha/
│   ├── FC212012-Sajani/
│   ├── FC212017-Dasun/
│   ├── FC212030_Mudhitha/
│   ├── FC212034_Kishgintharaam/
│   └── FC212036_Dileesha/
│
├── src/
│   ├── FC212009_Darsha/
│   ├── FC212012-Sajani/
│   ├── FC212017-Dasun/
│   ├── FC212030_Mudhitha/
│   ├── FC212034_Kishgintharaam/
│   └── FC212036_Dileesha/
│
├── tests/
│   └── test.py

````


## ⚙️ Prerequisites

- [Visual Studio Code](https://code.visualstudio.com/)  
- [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)  
- [Docker](https://www.docker.com/)  


## 🚀 Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/ddasunsandeepa/laptop_price_predictor_project.git
cd laptop_price_predictor_project
````

### 2️⃣ Open in VS Code

* Open the project folder in VS Code
* Click **"Reopen in Container"** when prompted
* VS Code will build the container image using `.devcontainer.json` and launch the development environment

### 3️⃣ Install dependencies inside the container

```bash
./Miniforge3-Linux-x86_64.sh
conda env create -f environment.yml
conda activate laptop-predict
```

### 4️⃣ Run the Flask app

```bash
cd app
python app.py
```

* Open your browser at [http://localhost:5000](http://localhost:5000) to use the laptop price predictor

## 🧩 How It Works

* All dependencies are installed with **Conda/Mamba**
* Data cleaning, exploration, and preprocessing are in `notebooks/`
* Model training, validation, and utility scripts are in `src/`
* The Flask app in `app/` loads pre-trained `.pkl` models to make predictions
* Original datasets and artifacts are in `data/`

## 🧪 Development Workflow

* **Explore & preprocess data:** Use notebooks in `notebooks/`
* **Train, validate, and test models:** Run scripts in `src/`
* **Deploy & test Flask app:** Use `app/`

## ✅ Next Steps

* Add automated unit tests in `tests/`
* Add CI/CD using GitHub Actions
* Deploy the Flask app to **Heroku, Render**, or similar

## 👨‍💻 Authors

* **Dasun Sandeepa** — [Dsk.sandeep.987@gmail.com](mailto:Dsk.sandeep.987@gmail.com)
* **Kishgintharaam Sathannthan** — [kishgi1234@gmail.com](mailto:kishgi1234@gmail.com)
* * **Gayana Sajani** — [gpgsajani@gmail.com](mailto:gpgsajani@gmail.com)

## 📜 License

This project is open-source. Add your preferred license here (e.g., MIT).

---

**Happy Predicting! 🚀**

