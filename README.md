# Obesity Classification using Azure Machine Learning and Databricks

This project demonstrates an end-to-end Machine Learning pipeline for predicting obesity levels based on demographic and biometric data such as Age, Gender, Height, Weight, and BMI.  
The solution integrates Azure Machine Learning, Databricks Lakeflow Jobs, and Python to enable automated training, deployment, and real-time inference in a cloud environment.

---

## Project Overview

The goal of this project is to classify individuals into four categories:
- Underweight  
- Normal weight  
- Overweight  
- Obese  

The project covers the complete lifecycle of model development:
1. Data ingestion and preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Model training and evaluation  
4. Model registration and deployment in Azure ML  
5. Pipeline orchestration using Lakeflow Jobs in Databricks  

---

## Technology Stack

| Category | Tools and Frameworks |
|-----------|----------------------|
| Cloud Platform | Azure Machine Learning, Azure Databricks, Azure Blob Storage |
| Language | Python |
| Machine Learning | Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib |
| Workflow Automation | Lakeflow Jobs, Azure ML Pipelines, Compute Clusters (AmlCompute) |
| Version Control | Git, GitHub |
| Deployment | Azure ML Endpoints |

---

## Machine Learning Pipeline Architecture

1. **Data Ingestion:** CSV dataset uploaded to Azure Blob Storage.  
2. **Preprocessing:** Missing value treatment, encoding categorical variables, feature scaling.  
3. **Modeling:** Decision Tree Classifier trained and validated on Azure ML compute cluster.  
4. **Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix.  
5. **Deployment:** Model registered and deployed for real-time prediction.  
6. **Orchestration:** Automated execution and monitoring using Lakeflow Jobs.

---

## Results

| Metric | Value |
|---------|-------|
| Accuracy | 93.4% |
| Precision (macro/micro) | 0.92 / 0.93 |
| Recall (macro/micro) | 0.91 / 0.93 |

The similarity between macro and micro metrics indicates balanced performance across all weight categories.

---

## Dataset

- **Source:** Custom dataset representing obesity levels and related attributes.  
- **Features:** Age, Gender, Height, Weight, BMI  
- **Target Variable:** Label (Underweight, Normal, Overweight, Obese)

---

## Deployment Details

- Workspace: BaharML-Canada  
- Resource Group: databricks-lab-rg  
- Compute Cluster: cpu-cluster  
- Pipeline ID: febf487e-a1e2-4f8b-92e7-02f7f46a54fd  
- Experiment Name: ObesityPrediction_Run  

Example execution in Azure ML:

```python
from azureml.core import Experiment
from azureml.pipeline.core import PublishedPipeline

published_pipeline = PublishedPipeline.get(ws, id=pipeline_id)
experiment = Experiment(workspace=ws, name="ObesityPrediction_Run")
run = experiment.submit(published_pipeline)
run.wait_for_completion(show_output=True)
```

---

## Integration with Databricks Lakeflow Jobs

Lakeflow Jobs are used to orchestrate the workflow and automate:
- Data preparation and validation  
- Model retraining and deployment  
- Periodic monitoring and retriggering of pipelines  

This approach ensures scalability, reproducibility, and adherence to MLOps best practices.

---

## Repository Structure

```
Obesity-Classification/
│
├── obesity_pipeline.py          # Main pipeline code
├── Obesity Classification.csv   # Dataset
├── CloudProject_Bahar.pptx      # Presentation slides
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
└── LICENSE                      # MIT License
```

---

## Author

**Bahar Almasi**  
Toronto, Canada  
Data Science and Analytics | Cloud Machine Learning | Azure ML | Databricks  
LinkedIn: [linkedin.com/in/bahar-almasi](https://linkedin.com/in/bahar-almasi)  
GitHub: [github.com/Bahar15984](https://github.com/Bahar15984)

---


## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.
