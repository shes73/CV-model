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

![image](https://github.com/user-attachments/assets/506a8ae8-8cf6-41b5-a480-36d048a51777)


*The project was completed as part of the final work for the course Machine Learning (NSU, Department of Mechanics and Mathematics, Prof. M.S. Tarkov)*
