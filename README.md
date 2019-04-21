## Information Retrieval
### Description
A **Text Mining** on Mini Newsgroups Dataset with **Feature Extraction**, **Feature Selection**, **Classification**, and **Clustering** algorithms (Information Retrieval Spring 2019)

### Instructions
To run the scripts locally:
1. Clone this repository. **NOTE:** Scripts are written with **Python 3.6.x.**
2. Navigate to the **scripts** folder and run ```pip install requirements.txt``` to install dependencies. 
3. For complete documentation, refer [to this document](https://github.com/Joeyipp/text-mining/blob/master/documentation/Design_Documentation.pdf).

### System Architecture
![Sample](https://github.com/Joeyipp/text-mining/blob/master/documentation/Design_Flowchart.png)

### Part 1: Feature Extraction
> ```python3 feature-extract.py mini_newsgroups feature_definition_file class_definition_file training_data_file```

![Sample](https://github.com/Joeyipp/text-mining/blob/master/images/feature_extract.PNG)

### Part 2: Classification
> ```python3 classification.py```

![Sample](https://github.com/Joeyipp/text-mining/blob/master/images/classification.png)

### Part 3: Feature Selection
> ```python3 feature_selection.py```

### Results
![Sample](https://github.com/Joeyipp/text-mining/blob/master/images/chi_squared.png)
![Sample](https://github.com/Joeyipp/text-mining/blob/master/images/mutual_information.png)

### Part 3: Clustering
> ```python3 clustering.py```

### Results
![Sample](https://github.com/Joeyipp/text-mining/blob/master/images/silhouette_coefficient.png)
![Sample](https://github.com/Joeyipp/text-mining/blob/master/images/normalized_mutual_information.png)

### References
* [Matplotlib](https://matplotlib.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/)
