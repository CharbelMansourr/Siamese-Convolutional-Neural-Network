# Siamese-Convolutional-Neural-Network

Unlike traditional CNNs that take an input of 1 image to generate a one-hot vector suggesting the category the image belongs to, the Siamese network takes 
in 2 images and feeds them into 2 CNNs with the same structure. 
The output would be merged together, in this case through their absolute differences, and feed into fully connected layers to output one number 
representing the similarity of the two images.A larger number implies that the two images are more similar.  
The goal of this code is to implement a siamese CNN and apply it on the newdata set given.
