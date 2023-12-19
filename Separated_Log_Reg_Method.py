"""
This file is meant to train and test a five separate binary logistic regressions so the classification with the highest probability can be selected.
The accuracy of the classifications are not calculated in this file.

Written by Grange Simpson
Version: 2023.12.18

Usage: Run the file.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

"""
Class used for defining, training, and running 5 logistic regression classifiers at once.
"""
class LogRegClass:
    def __init__(self):
        # Datasets used for training the logistic regression classifiers.
        self.trainDataX = 0
        self.trainDataY = 0
        # mean and std for the z score
        self.trainDataMean = pd.DataFrame()
        self.trainDataStd = pd.DataFrame()
        # dataframe for saving all y data for each binary classifier.
        self.multiClassYData = pd.DataFrame()
        # logistic regression variables for 
        self.LR_DH1_2 = 0
        self.LR_DH2_3 = 0
        self.LR_DH3_3 = 0
        self.LR_DH4_3 = 0
        self.LR_DH5_2 = 0

    """
    Method for normalizing, splitting, and then training all five binary classifiers.
    """
    def train_LR_model(self, inputXData, inputYData):
        # Normalize incoming data
        self.z_score_data(inputXData, inputYData)
        # Split incoming y data into multiple classifications
        self.multi_class_split_selection()
        # Train each logistic regression from the split y data
        self.train_multiclass_log_reg()

    """
    All data needs to be z score normalized 
    """
    def z_score_data(self, logregXData, logregYData):
        self.trainDataX = logregXData
        self.trainDataY = logregYData

        self.trainDataMean = logregXData.mean(axis=0)
        self.trainDataStd = logregXData.std(axis=0)

        # Remove NaN values from the dataset since they break the logistic regression and then zscore normalize training data
        for column in logregXData.columns:
            self.trainDataX[column] = self.trainDataX[column].replace(np.nan, self.trainDataX[column].mean())
            self.trainDataX[column] = (self.trainDataX[column] - np.mean(self.trainDataX[column])) / np.std(self.trainDataX[column])

    """
    Set up the y data for all 5 logistic regressions to be trained on.
    All data that was preselected for the given classification is saved as its value, then all other data is set to zero so there are only binary classifications.
    """
    def multi_class_split_selection(self):
        y_data = pd.DataFrame()
        
        # First classification
        y_data['DH1_2_Class'] = self.trainDataY
        y_data['DH1_2_Class'] = np.where(y_data['DH1_2_Class'] != 0, 1, y_data['DH1_2_Class'])

        # Second classification
        y_data['DH2_2_Class'] = self.trainDataY
        y_data['DH2_2_Class'] = np.where(y_data['DH2_2_Class'] != 1, 0, y_data['DH2_2_Class'])

        # Third classification
        y_data['DH3_2_Class'] = self.trainDataY
        y_data['DH3_2_Class'] = np.where(y_data['DH3_2_Class'] != 2, 0, y_data['DH3_2_Class'])

        # Fourth classification
        y_data['DH4_2_Class'] = self.trainDataY
        y_data['DH4_2_Class'] = np.where(y_data['DH4_2_Class'] != 3, 0, y_data['DH4_2_Class'])

        # Fifth classification
        y_data['DH5_2_Class'] = self.trainDataY
        y_data['DH5_2_Class'] = np.where(y_data['DH5_2_Class'] != 4, 0, y_data['DH5_2_Class'])

        self.multiClassYData = y_data

    """
    Method for setting up a logistic regression for each classification.
    """
    def create_log_reg(self, XData, YData):
        logReg = LogisticRegression(random_state = 6, max_iter = 200)
        logReg.fit(XData, YData)

        return logReg

    """
    Train all five of logistic regressions for binary classification of the given class
    """
    def train_multiclass_log_reg(self):
        # First class
        self.LR_DH1_2 = self.create_log_reg(self.trainDataX, self.multiClassYData['DH1_2_Class'])
        
        # Second class
        self.LR_DH2_3 = self.create_log_reg(self.trainDataX, self.multiClassYData['DH2_2_Class'])

        # Third class
        self.LR_DH3_3 = self.create_log_reg(self.trainDataX, self.multiClassYData['DH3_2_Class'])

        # Fourth class
        self.LR_DH4_3 = self.create_log_reg(self.trainDataX, self.multiClassYData['DH4_2_Class'])

        # Fifth class
        self.LR_DH5_2 = self.create_log_reg(self.trainDataX, self.multiClassYData['DH5_2_Class'])

    """
    Run the testing data through all 5 of the binary classifiers then save to one dataframe to return.
    """
    def run_input_through_all_trained_logregs(self, inputDiscrWaveform):
        # Z scoring according to training data mean and std
        inputDiscrWaveform = inputDiscrWaveform - self.trainDataMean.to_numpy()
        inputDiscrWaveform = inputDiscrWaveform / self.trainDataStd.to_numpy()

        # Take output probabilities from each input dataframe and add to one big df
        all_probabilities = pd.DataFrame()

        # Create dataframe out of the prediction probabilities for all 5 binary classifiers
        all_probabilities =  pd.DataFrame(self.LR_DH1_2.predict_proba(inputDiscrWaveform))
        all_probabilities = pd.concat([all_probabilities, pd.DataFrame(self.LR_DH2_3.predict_proba(inputDiscrWaveform))], axis = 1)
        all_probabilities = pd.concat([all_probabilities, pd.DataFrame(self.LR_DH3_3.predict_proba(inputDiscrWaveform))], axis = 1)
        all_probabilities = pd.concat([all_probabilities, pd.DataFrame(self.LR_DH4_3.predict_proba(inputDiscrWaveform))], axis = 1)
        all_probabilities = pd.concat([all_probabilities, pd.DataFrame(self.LR_DH5_2.predict_proba(inputDiscrWaveform))], axis = 1)

        return all_probabilities

# Import the all training data.
stats_df = pd.read_table('relabeled_Tbl_500.txt', sep = '\t', header = 0)

# Import the metrics to be used based off of the decision tree outputs
FS = pd.read_table('FSfinal.txt', sep = '\t', header = None)

# X data and y data
X_stats_df = stats_df.drop(columns=['DH','HS_loc','TO_loc','Start_step','ptn','days_to_healed','N_to_healed','Step_label','Weight_lb'])
y = stats_df.DHC

DT_train = stats_df[FS[0].to_list()]
y_train = stats_df.DHC

Test_df = pd.read_table('Table_500_5g.txt', sep = '\t')

DT_test = Test_df[FS[0].to_list()]
y_test = Test_df.DHC

# Only train the log reg one time so that a mean of a mean isn't taken.
firstMethod = LogRegClass()
firstMethod.train_LR_model(DT_train, y_train)

# Decision tree x and y testing data
DT_X_waveform_stats_df = DT_test[FS[0].to_list()]

DT_Y_waveform_stats_df = y_test

# Remove NaN values from the testing dataset and replacing them with the mean of that column so the dataset isn't skewed.
for column in DT_X_waveform_stats_df.columns:
    DT_X_waveform_stats_df[column] = DT_X_waveform_stats_df[column].replace(np.nan, DT_X_waveform_stats_df[column].mean())

# Reseting the index of the testing dataframe
DT_X_waveform_stats_df.index = np.arange(0, len(DT_X_waveform_stats_df.index))

# Running the test data through the trained logistic regression.
output_classifications = firstMethod.run_input_through_all_trained_logregs(DT_X_waveform_stats_df)

# Changing to dataframe structure so data can be saved
check_df = pd.DataFrame(output_classifications)

print("Finished, saving data.")
check_df.to_csv('sep_13_run_auc_roc.txt', sep = '\t')

