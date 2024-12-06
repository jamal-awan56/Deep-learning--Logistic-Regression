# Deep-learning--Logistic-Regression
1. Introduction to PyTorch Introduces PyTorch's core features, focusing on tensors and their operations. Sets the foundation for using PyTorch to build and train machine learning models.

2. Dataset: MNIST Utilizes the MNIST dataset, a benchmark dataset of 28x28 grayscale images of handwritten digits (0-9). Demonstrates how to load the dataset using PyTorch's torchvision.datasets. Splits the data into training and testing sets. Applies necessary transformations, such as normalizing the pixel values to a range between 0 and 1.

3. Model: Logistic Regression Implements a logistic regression model using PyTorch's torch.nn module. The model is defined with a single linear layer mapping the 784 input features (flattened 28x28 image) to 10 output classes. Applies the softmax function to convert raw scores into probabilities. Explains the mathematical foundation of logistic regression and its application in multi-class classification.

4. Loss Function and Optimizer Uses Cross-Entropy Loss as the loss function to handle multi-class classification. Selects Stochastic Gradient Descent (SGD) as the optimization algorithm, with a defined learning rate.

5. Training the Model The notebook trains the logistic regression model over several epochs: Performs forward propagation: Passes input data through the model. Calculates the loss using the predictions and true labels. Executes backpropagation to compute gradients. Updates model weights using the optimizer. Includes checkpoints for monitoring the loss during training.

6. Model Evaluation Evaluates the model on the test set to measure its performance. Calculates the final accuracy, achieving 80%, which is a reasonable result for a simple logistic regression model on MNIST. Discusses the model's strengths and limitations, including potential improvements through more complex architectures or hyperparameter tuning.

7. Visualizations Visualizes: A sample of the MNIST images with their predicted and true labels. Training loss over epochs to demonstrate convergence. 8. Results and Insights

Results: The logistic regression model achieved 80% accuracy on the test set. Highlights that while logistic regression provides a good baseline, more advanced models (e.g., neural networks) could further improve accuracy. Provides insights into the importance of data preprocessing and model selection in achieving good performance.
