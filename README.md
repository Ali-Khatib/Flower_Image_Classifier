Dataset The dataset used is Oxford Flowers 102, which contains images of 102 different flower species. It is loaded using TensorFlow Datasets (TFDS), and the dataset is split into training, validation, and test sets.

Model A simple feedforward neural network is used for this classification task. The model architecture consists of:

A flatten layer to convert images into a 1D array.

A dense layer with ReLU activation for feature extraction.

A dropout layer to avoid overfitting.

A final dense layer with softmax activation to predict the class.

Usage Load the dataset using TensorFlow Datasets.

Split the dataset into training, validation, and test sets.

Preprocess the images to resize them and normalize the pixel values.

Train the model for 10 epochs and plot the training and validation accuracy/loss.

Evaluate the model on the test set.

Save the trained model for future use.

Model Training & Evaluation Training: The model is trained on the training dataset and validated using the validation set.

Evaluation: The model's performance is evaluated on the test set, with test accuracy and loss displayed.

Plots: Training and validation accuracy and loss are plotted to visualize the model's progress.

Example Outputs: The model will output training and validation accuracy/loss graphs and the final evaluation results on the test set.
