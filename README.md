# Neural_Network_Charity_Analysis
Implement neural networks using the TensorFlow platform in Python.  Use the features in the provided dataset to help create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup, a non profit philanthrophic foundation.

## Neural_Network_Charity_Analysis:
Alphabet Soup is a non profit philanthropic foundation and Beks is a data scientist and programmer there. The task is to help Aplhabet Soup and Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup based on which Alphabet Soup will decide which organizations should recieve donations.

From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years and the problem seems too complex for statistical models being used in the company so its deciced to implement neural networks using the TensorFlow library in Python.

## Resources:
* Data: charity_data.csv
* Software: Jupyter Notebook 6.4.8 , Python 3.7.11 

## Results:
#### The code for deliverable 1 and deliverable 2 can be viewed in [AlphabetSoupCharity.ipynb]()

## The steps involved in deliverable 1 included 
* Dropping The EIN and NAME columns.

![EIN_NAME]()

* The columns with more than 10 unique values being grouped together.
![unique_values]()

![app]()

![class]()


* The categorical variables encoded using one-hot encoding.

![hot]()

* The preprocessed data split into features and target arrays. The preprocessed data split into training and testing datasets. The numerical values have been standardized using the StandardScaler() module.

![merge]()

![split_train]()

#### Steps involved in deliverable 2 included:
At the end of deliverable one we can see that the number of input features is 43. Adopting the neural network model using Tensorflow Keras we can see that the basic neural network model for 2 inputs instead of 43 would look would have one input layer, a hidden layer which had 2-3 times neurons as that of input and the output layer. The data from each layer is processed using the activation functions that can be relu, tanh, sigmoid etc. 
depending upon the requirements of the model.

![neural]()

Our code performs the following steps:
* Since we have **43** inputs we need at least **2** layers , and **80** neurons in layer1 and **30** in layer2 and activation functions would be **relu** for input and hidden layers and **sigmoid** for output as it is binary classification prediction.

![model1]()

* The output of the model’s loss and accuracy is:

![compile]()

* The model's weights are saved every 5 epochs in [checkpoints]()

![call_back]()

* The results are saved to an HDF5 file [AlphabetSoupCharity.h5]() and the model accuracy is increased to 72.50 %.

![save]()


#### The results for deliverable 3 can be viewed as:
* code in [AlphabetSoupCharity_Optimization.ipynb]()
* checkpoints in [checkpoints_Op]()
* saved h5 file in [AlphabetSoupCharity_Op.h5]()

#### Data Preprocessing
1. The variables considered as target for the model is IS_SUCCESSFUL feature.
2. Variable(s) are considered to be the features are :
    - NAME—Identification column
    - APPLICATION_TYPE—Alphabet Soup application type
    - AFFILIATION—Affiliated sector of industry
    - CLASSIFICATION—Government organization classification
    - USE_CASE—Use case for funding
    - ORGANIZATION—Organization type
    - STATUS—Active status
    - INCOME_AMT—Income classification
    - SPECIAL_CONSIDERATIONS—Special consideration for application
    - ASK_AMT—Funding amount requested
3. EIN—Identification column is redundant so it should be removed from the input data.
4. Compiled, trianed and evaluated using loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"].
5. Finally, after 2 rounds of optimizations where the inputs were decreased from 43 to 40 but the model performance got worse so the number of inputs was increased to 140 and then to 146. Due to higher number of inputs, hidden layers
were increased to 4, and number of neurons were increased to 250, 130, 45, 15 for hidden layer 1 to 4 respectively. Activation layers were **relu** for the input and the 4 hidden layers and **sigmoid** for output layer. 

The target model performance was acheived and the final performance was **77.19 %**

![optimum]()

6. The following steps were taken to increase the model performance.
    1. Increased the number to hidden layers to **3** , total parameters became **10086** with neurons as follows:
        * hidden_nodes_layer1 = 110
        * hidden_nodes_layer2 = 45
        * hidden_nodes_layer3 = 10

    The accuracy increased to 72.93. But below 75%.
    2. The **SPECIAL_CONSIDERATIONS_N** row was dropped as it was redundant. Activation function for hidden and input layers was changed to **tanh**.The inputs reduced from 43 to 40 The model parameters were as follows:
        * hidden_nodes_layer1 = 110 neurons
        * hidden_nodes_layer2 = 45 neurons
        * hidden_nodes_layer3 = 10 neurons

    The accuracy dropped to 72.59. 

    3. Reducing the number of inputs did not increase the accuracy so data given to the model seemed to be insuffient for better predictions so:
        * Only EIN was dropped and NAME column was restored.
        * Binning threshold was reduced to increase the input variables, 
            - application_counts < 100 instead of 500
            - name_counts <25 
        * This increased the number of inputs from 43 to **140**. With increase in inputs 4 hidden layers were used as follows:
            - hidden_nodes_layer1 = 150 neurons
            - hidden_nodes_layer2 = 75 neurons
            - hidden_nodes_layer3 = 20 neurons
            - hidden_nodes_layer4 = 10 neurons
            - There were a total of 34216 parameters.
            - Since the activation function change from relu to tanh didn't cause much changes the activation function was kept as **relu**. 

    The model efficiency increased to 76.94 % and a loss of 47.85 %

    4. Finally, 
        * Binning threshold was reduced to increase the input variables, 
            - class_counts < 100 instead of 1800
        * This increased the inputs from 140 to **146**. With increase in inputs 4 hidden layers were used as follows:
            - hidden_nodes_layer1 = 250 neurons
            - hidden_nodes_layer2 = 130 neurons
            - hidden_nodes_layer3 = 45 neurons
            - hidden_nodes_layer4 = 15 neurons
            - There were a total of 75981 parameters.
            - Since the activation function change from relu to tanh didn't cause much changes the activation function was kept as **relu**. 

    The model efficiency increased to 77.19 % and a loss of 50.10 %. Since, the gain in efficiency was not very drastic and loss was increasing. It was the optimal model choosen.


## Summary:
1. Finally, the model achieves 77.19 % accuracy against the accuracy of 72.50 achieved after deliverable 2.

| Property | Accuracy |
| --- | --- |
| Before optimization     |72.50    |
| First Attempt	          |72.93    |	
| Second Attempt          |72.59    |	
| Third Attempt           |76.94    |	
| Final Attempt           |77.19    |
			

![optimum]()

2. Recommendation on a different model would be a **Random Forest Clasisfier**, because:
    - The Random forest classifier works very well with tabular data.
    - There are only 34000 rows in this csv file so Random forest classifier can handle it easily while ANN involves much more complexity.
    - It can handle thousands of input variables without variable deletion and in this case where data seems to be less it seems a good choice to keep all the data.
