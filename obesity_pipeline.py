#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import plot_tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# In[27]:


from azureml.core import Workspace

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id)


# In[28]:


from azureml.core import Workspace

ws = Workspace(
    subscription_id="252c4afd-7636-4d18-b1b4-11d3afbe522d",
    resource_group="BaharCanada",
    workspace_name="BaharML-Canada"
)

print("Connected to Workspace:", ws.name)


# In[29]:


get_ipython().system('pip install azureml-core azureml-pipeline-core --upgrade')


# In[30]:


get_ipython().system('pip install azureml-sdk')


# In[31]:


from azureml.pipeline.core import PublishedPipeline

pipelines = PublishedPipeline.list(ws)
for p in pipelines:
    print(f"Name: {p.name} | ID: {p.id}")


# In[32]:


from azureml.core import Experiment
from azureml.pipeline.core import PublishedPipeline

# Replace with actual ID from previous step
pipeline_id = "your-pipeline-id-here"  

published_pipeline = PublishedPipeline.get(ws, id=pipeline_id)

# Create an experiment to track this run
experiment = Experiment(ws, "ObesityPrediction_Run")

# Submit the pipeline
run = experiment.submit(published_pipeline)
run.wait_for_completion(show_output=True)


# In[33]:


pipeline_id = "febf487e-a1e2-4f8b-92e7-02f7f46a54fd"


# In[34]:


from azureml.core import Experiment
from azureml.pipeline.core import PublishedPipeline

# Get the pipeline using the ID
published_pipeline = PublishedPipeline.get(ws, id=pipeline_id)

# Create an experiment to track the pipeline run
experiment = Experiment(workspace=ws, name="ObesityPrediction_Run")

# Submit the pipeline
run = experiment.submit(published_pipeline)
run.wait_for_completion(show_output=True)


# In[37]:


from azureml.core import Workspace

ws = Workspace.from_config()
print(ws.name)


# In[38]:


get_ipython().system('pip install azureml-pipeline-core')


# In[39]:


from azureml.core import Workspace

ws = Workspace.from_config()

print("Connected to workspace:", ws.name)


# In[40]:


from azureml.pipeline.core import PipelineEndpoint


pipelines = PipelineEndpoint.list(ws)

for p in pipelines:
    print("Name:", p.name, "| ID:", p.id)


# In[41]:


get_ipython().system('pip install azureml-pipeline-core --upgrade')


# In[42]:


from azureml.core import Workspace

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, sep='\n')


# In[ ]:


get_ipython().system('pip3 install azureml-pipeline-core --upgrade')


# In[43]:


from azureml.pipeline.core import PipelineEndpoint
print("✅ Module imported successfully!")


# In[44]:


import os
import requests
import tempfile
import azureml.core
from azureml.core import Workspace, Experiment, Datastore

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)


# In[45]:


from azureml.pipeline.steps import PythonScriptStep

print("Pipeline SDK-specific imports completed")


# In[46]:


ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')

# Default datastore
def_blob_store = ws.get_default_datastore() 
# The following call GETS the Azure Blob Store associated with your workspace.
# Note that workspaceblobstore is **the name of this store and CANNOT BE CHANGED and must be used as is** 
def_blob_store = Datastore(ws, "workspaceblobstore")
print("Blobstore's name: {}".format(def_blob_store.name))


# In[47]:


# download data file from remote
response = requests.get("https://dprepdata.blob.core.windows.net/demo/Titanic.csv")
titanic_file = os.path.join(tempfile.mkdtemp(), "Titanic.csv")
with open(titanic_file, "w") as f:
    f.write(response.content.decode("utf-8"))
# get_default_datastore() gets the default Azure Blob Store associated with your workspace.
# Here we are reusing the def_blob_store object we obtained earlier
def_blob_store.upload_files([titanic_file], target_path="titanic", overwrite=True)
print("Upload call completed")


# In[48]:


cts = ws.compute_targets
for ct in cts:
    print(ct)


# In[49]:


from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

aml_compute_target = "cpu-cluster"
try:
    aml_compute = AmlCompute(ws, aml_compute_target)
    print("found existing compute target.")
except ComputeTargetException:
    print("creating new compute target")
    
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2",
                                                                min_nodes = 1, 
                                                                max_nodes = 4)    
    aml_compute = ComputeTarget.create(ws, aml_compute_target, provisioning_config)
    aml_compute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
print("Azure Machine Learning Compute attached")


# In[50]:


import pandas as pd

df = pd.read_csv("Logs/baharalmasi1/ObesityClassification.csv")
df.head()


# In[51]:


df = pd.read_csv("Users/baharalmasi1/SecurityUpdatesDiagnostics/ObesityClassification.csv")
df.head()


# In[52]:


import os

for root, dirs, files in os.walk("Users"):
    for name in files:
        print(os.path.join(root, name))


# In[53]:


import os

for root, dirs, files in os.walk("/mnt"):
    for name in files:
        if "Obesity" in name:
            print(os.path.join(root, name))


# In[54]:


import pandas as pd

df = pd.read_csv("/mnt/batch/tasks/shared/LS_root/mounts/clusters/baharalmasi1/code/Users/baharalmasi/CSVfile/Obesity Classification.csv")
df.head()


# In[55]:


import pandas as pd

# Load the dataset
df = pd.read_csv("/mnt/batch/tasks/shared/LS_root/mounts/clusters/baharalmasi1/code/Users/baharalmasi/CSVfile/Obesity Classification.csv")

# Quick overview of the data
print(df.head())

# Statistical summary
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Data types of each column
print(df.dtypes)

# Unique values in categorical columns
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].unique()}")  # ✅ This line is now fixed


# In[56]:


# Class distribution (target variable)
print("🔹 Class distribution:")
print(df['Label'].value_counts())


# In[57]:


# Quick overview
print("🔹 Shape of the dataframe:", df.shape)
print("🔹 First few rows:")
display(df.head())

# Statistical summary
print("\n🔹 Statistical summary:")
display(df.describe())

# Missing values
print("\n🔹 Missing values per column:")
display(df.isnull().sum())

# Data types
print("\n🔹 Column data types:")
print(df.dtypes)

# Unique values in categorical columns
print("\n🔹 Unique values in categorical columns:")
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].unique()}")


# In[58]:


# Class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Label', data=df)
plt.title("Distribution of Obesity Classes")
plt.xlabel("Obesity Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[59]:


# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[60]:


numeric_cols = ['Age', 'Height', 'Weight', 'BMI']
df[numeric_cols].hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.suptitle("Distribution of Numeric Features", y=1.02)
plt.show()


# In[61]:


for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Label', y=col, data=df)
    plt.title(f"{col} by Obesity Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[62]:


le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])
print("Encoded Labels:", le.classes_)  # You can use this later to decode predictions


# In[63]:


categorical_cols = df.select_dtypes(include='object').columns.tolist()

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("Shape after one-hot encoding:", df.shape)


# In[64]:


from sklearn.preprocessing import StandardScaler

numeric_cols = ['Age', 'Height', 'Weight', 'BMI']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# In[65]:


from sklearn.model_selection import train_test_split

# Set the target column
target_col = 'Label'  # change this if your target has a different name

# Split features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)


# In[66]:


from sklearn.tree import DecisionTreeClassifier

# Create the model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)


# In[67]:


# Predict on the test set
y_pred_dt = dt_model.predict(X_test)


# In[68]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

# Confusion matrix (optional)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[69]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Convert class names to strings
class_names = [str(c) for c in dt_model.classes_]

# Plot the tree
plt.figure(figsize=(20,10))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=class_names)
plt.title('Decision Tree Visualization')
plt.show()


# In[70]:


print(classification_report(y_test, y_pred_dt))


# #### Label's Classes 
# 0 → Normal Weight
# 1 → Obese
# 2 → Overweight
# 3 → Underweight
# 

# 
