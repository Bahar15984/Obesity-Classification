# 🩺 Obesity Classification (Azure ML + Python)

Predicts **obesity level** from basic features (Age, Gender, Height, Weight, BMI) and demonstrates an **end-to-end ML pipeline** on **Azure Machine Learning** with real-time inference.

## 🚀 Highlights
- End-to-end pipeline: data prep → train → evaluate → register → deploy (real-time endpoint).
- Reproducible runs via Azure ML SDK (experiment: `ObesityPrediction_Run`).
- Consistent macro/micro precision/recall across weight classes (robust class performance).

## 🧠 Problem & Labels
Multi-class classification with 4 classes:
1. Underweight
2. Normal Weight
3. Overweight
4. Obese

## 📦 Project Structure
```
Obesity-Classification/
├── README.md
├── obesity_pipeline.py                # Training + deployment pipeline (add your file here)
├── Obesity Classification.csv         # Dataset (add your file here)
├── CloudProject_Bahar.pptx            # Slides (optional, add your file here)
├── images/
│   └── model_metrics.png              # (optional) add plots/screens
└── notebooks/
    └── Obesity_Training.ipynb         # (optional) exploratory notebook
```

> ⚠️ **Note:** This repo contains no secrets. Do **not** commit any keys or connection strings. Use Azure Key Vault for secrets.

## ⚙️ Tech Stack
- Python (pandas, scikit-learn)
- Azure Machine Learning SDK
- Optional: Databricks for data prep
  
## 🧪 Quickstart
1. Clone the repo and add your files (`obesity_pipeline.py`, dataset, slides).
2. Create and activate a virtual env:
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt  # (create as needed)
   ```
3. Run your pipeline script:
   ```bash
   python obesity_pipeline.py
   ```

## 📝 Results (example summary)
- Macro vs micro precision/recall are close → consistent performance across classes.
- Pipeline executed successfully (`ObesityPrediction_Run`) and completed without errors.

## 🧩 Next Steps
- Add confusion matrix & classification report to `/images` and reference them here.
- Add a `requirements.txt` with exact versions used.
- Publish a short demo GIF or screenshot of Azure ML endpoint test.

## 👩🏻‍💻 Author
**Bahar Almasi**
- LinkedIn: https://www.linkedin.com/
- GitHub: https://github.com/

## 📄 License
MIT
