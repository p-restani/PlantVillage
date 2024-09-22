# PlantVillage Project

This project uses Convolutional Neural Networks (CNNs) for plant disease classification, based on the **PlantVillage** dataset.

## Instructions

### 1. Clone the Repository
Clone the repository to your local machine or save it in your Google Drive:

```bash
git clone https://github.com/p-restani/PlantVillage.git
```


### 2. Download the Dataset
Download the PlantVillage dataset from this [link](https://www.kaggle.com/datasets/emmarex/plantdisease). After downloading, make sure to correct the dataset path in the code to reflect the location of the dataset.

````bash
data_dir = 'your-path-to-dataset/data/PlantVillage'
````

### 3. Using Google Colab
If you are using Google Colab, follow these steps:

Upload the dataset to your Google Drive.
Connect your Google Drive to Colab by running the following command:

````bash
from google.colab import drive
drive.mount('/content/drive')

````


### 4. Run the Notebook

Once the dataset paths are updated, you can run the entire notebook to train the model:

1. Open the notebook file `PlantVillage.ipynb` in Google Colab or Jupyter Notebook.
2. Run each cell sequentially by clicking on the "Run" button or pressing `Shift + Enter`.
3. Make sure the dataset paths are correctly updated as shown in the instructions.


### 5. Run the Notebook from Saved Model Parameters
If you have already completed the model training (Parameter Tuning with Cross Validation) and saved the model parameters, you can load the saved model and continue from there. Follow these steps to proceed:

1. Open the notebook PlantVillage_Notebook.ipynb in Google Colab or Jupyter Notebook.
2. Re-run the imports and data preprocessing steps to ensure the dataset is reloaded and preprocessed. This includes resizing, normalizing, and applying any data augmentation.
3. Continue the training from "Training and Validation loss / F1 score over 20 epochs"
   
   
### 6. Run the Notebook from GradCAM Visualization

**Note**: This step assumes that the model parameters have been saved and the model training has been completed. If the training is not finalized, you need to first complete the training process before running GradCAM.

1. **Open the notebook** `PlantVillage.ipynb` in Google Colab or Jupyter Notebook.
2. **Re-run the imports and data preprocessing steps** to ensure the dataset is reloaded and preprocessed.
3. Re-run the cell with CNN architecture definition before loading the model:

````bash
class TunableCNN(nn.Module):
    def __init__(self, num_classes, conv1_out_channels, conv2_out_channels, dropout_rate):
        super(TunableCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, conv1_out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_out_channels)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(conv2_out_channels * 1 * 1, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

````

4. Re-run the cell to load the CSV file with the results of hyperparameter tuning:

````bash
# Load the CSV file with a relative path
results_df = pd.read_csv('hyperparameter_results.csv')

# Display the dataframe to understand its structure
results_df.head()
````

5. Re-run the cell to extract the best parameters and their corresponding metrics:

````bash
# Extract the best parameters and their corresponding metrics
best_result = results_df.loc[results_df['f1_score'].idxmax()]
best_params = best_result.to_dict()
best_f1 = best_params['f1_score']

print(f'Best F1 Score: {best_f1}')
print(f'Best Params: {best_params}')
````

6. Continue by running the remaining GradCAM cells in the notebook.


