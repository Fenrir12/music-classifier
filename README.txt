
clone music classifier and competition datasets from :
https://github.com/Fenrir12/music-classifier.git

cd into music-classifier

Download song data from : https://www.kaggle.com/c/music-genre-classification/data

Move the content of song_data.zip in the 'datasets' folder

cd into scripts folder

To run basic gaussian naive bayes :
python GaussianClassifier_v0.py <use_test or use_val>
	use_test: Will extract accuracy by taking a test set inside training set
	use_val: Will classify test set with unknown labels

To run knn classifier
	To use basic knn:
	python GaussianClassifier_v0.py <use_test or use_val>
		use_test: Will extract accuracy by taking a test set inside training set
		use_val: Will classify test set with unknown labels
		"Takes 15 seconds to compute one song"
	To use gaussian kernel knn:
	python GaussianClassifier_v1.py <use_test or use_val>
		use_test: Will extract accuracy by taking a test set inside training set
		use_val: Will classify test set with unknown labels
		"Takes 20 seconds to compute one song"

If using classifier on validation set, results will be available in test_labels.csv for gaussian and knn v0 and test_labels_v1.csv for knn v1
