# CV-model
An attempt to replicate the model from the article by Aysha Naseer and Ahmad Jalal
DOI: 10.1109/ICACS60934.2024.10473242

Briefly, the authors propose a computer vision algorithm based on a specific sequence of actions:
1. Image processing using a median filter
2. Image segmentation using a Gaussian Mixture Model
3. Feature extraction using HOG, MSER, and KAZE filters
4. Fusing of features in the MLP model

Since the article lacks technical details, some parameters and the architecture of the MLP model have been chosen by myself.

In the repo, you can find input data in the folder *"corel10k_10"*. It includes 10 classes – bike, bird, boat, building, car, deer, dog, elephant, flower, horse – from the Corel10K dataset. You can also find preprocessed and segmented images in the folders *"corel10k_preprocessed"* and *"corel10k_segmented"*, respectively. The results of feature extraction using the HOG, MSER, and KAZE filters are also presented in the *"extracted_features"* folder.

Results obtained using my implementation of the model:

Model Accuracy: 74.50%
Classification Report:
               precision    recall  f1-score   support

        bike       0.78      0.35      0.48        20
        bird       0.45      0.75      0.57        20
        boat       0.95      0.95      0.95        20
    building       0.70      0.95      0.81        20
         car       0.82      0.90      0.86        20
        deer       0.71      0.50      0.59        20
         dog       0.59      0.65      0.62        20
    elephant       0.92      0.60      0.73        20
      flower       0.87      1.00      0.93        20
       horse       0.94      0.80      0.86        20

    accuracy                           0.74       200
   macro avg       0.77      0.74      0.74       200
weighted avg       0.77      0.74      0.74       200

Confusion Matrix:
 [[ 7  1  0  2  4  3  2  0  1  0]
 [ 0 15  1  0  0  0  4  0  0  0]
 [ 0  1 19  0  0  0  0  0  0  0]
 [ 0  0  0 19  0  1  0  0  0  0]
 [ 0  0  0  2 18  0  0  0  0  0]
 [ 0  9  0  1  0 10  0  0  0  0]
 [ 1  2  0  1  0  0 13  0  2  1]
 [ 0  3  0  2  0  0  3 12  0  0]
 [ 0  0  0  0  0  0  0  0 20  0]
 [ 1  2  0  0  0  0  0  1  0 16]]

*The project was completed as part of the final work for the course Machine Learning (NSU, Department of Mechanics and Mathematics, Prof. M.S. Tarkov)*
