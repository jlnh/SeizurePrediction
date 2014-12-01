American Epilepsy Society Seizure Prediction Challenge
-------------------------------
**Description**

The code was written for the [American Epilepsy Society Seizure Prediction Challenge](https://www.kaggle.com/c/seizure-prediction). The software is written in Python. The standard numpy, scipy, scikit-learn and matplotlib packages are used extensively.

**Dependency**
  * Python 2.7
  * scikit learn-0.14.1
  * numpy-1.8.1
  * pandas-0.14.0
  * scipy
  * hickle (plus h5py and hdf5, see https://github.com/telegraphic/hickle for installation details)

**Hardware and Runtime**

The computations were done in a desktop with intel i7 quad core CPU and 12GB RAM. The total computational time is about 2 hours when the classifier is set to use all the CPU threads.

**How to Generate the Solution**
	
* Modify SETTINGS.json file and put the data in the data dir. Sample SETTINGS.json is given here
```
{
  "competition-data-dir": "seizure-data",
  "data-cache-dir": "data-cache",
  "submission-dir": "submissions",
  "figure-dir": "figure"
}
```
* Run predict.py
* Check the submission file in submissions directory and the analytical graphic PDF file in figure directory
