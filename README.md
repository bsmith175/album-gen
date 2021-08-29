### Album Cover Generation

# Introduction
Our project aims to use an Auxiliary Classifier Generative Adversarial Network to generate rich, expressive album cover artwork based on genre descriptors. The model is based on a paper titled Album Cover Generator from Genre Tags, which outlines a method for creating unique images that reflect the characteristics of its genre.

In this project, the generator model is an unsupervised learning problem, as we are trying to generate images rather than predicting anything. The discriminator model is a classification problem, as its job is to classify images based on genre tags. Part of what makes this problem interesting and difficult is that there is very high variation among album covers and subtle differences between genres.

# Methodology
The model is a Deep Convolutional Auxiliary Classifier Generative Adversarial Network. The generator consists of 1 fully-connected layer and 4 deconvolutional layers. The discriminator consists of 4 convolutional layers and 3 fully-connected layers. In the discriminator, the first dense layer predicts if the image was real or generated, the second dense layer predicts an image’s genre label, and the third dense layer predicts the image’s latent space variables. 

Because of the lack of labeled data, the model was pre-trained on a large dataset of 250,000 unlabeled images from the One Million Audio Cover Images for Research (OMACIR) dataset to produce album cover images irrespective of genre labels. Then, it was trained on the smaller dataset that we created using the Napster API of 9,000 album covers with genre labels. 

# Results
We evaluated the model with the Frechet Inception Distance (FID) between real and generated images, the discriminator’s accuracy on predicting class labels for real and generated images, and visual comparison. We reached an average FID of 1360 for the labeled data, and an average FID of 1546 for the unlabeled data. The discriminator classified genres with an accuracy of 94.6% for generated images, compared to an accuracy of 22.8% for real images. 
Example generated images:

Genre: jazz

![image](https://user-images.githubusercontent.com/52060846/131236554-70d8724b-245c-4cac-9a03-cff1927fc5ac.png) ![image](https://user-images.githubusercontent.com/52060846/131236577-9414cc83-b301-4fc0-ae3e-b6a084075e12.png)

Genre: Pop

![image](https://user-images.githubusercontent.com/52060846/131236586-51ccbf44-dfa7-440f-b9b2-98e77303acd4.png) ![image](https://user-images.githubusercontent.com/52060846/131236588-0b1e9627-1265-4cfb-abba-6b1e95ab8bca.png)

Genre: classical

![image](https://user-images.githubusercontent.com/52060846/131236594-49e533f8-1914-45a8-9606-dd64c962067e.png) ![image](https://user-images.githubusercontent.com/52060846/131236597-a36e2141-a3e9-4028-9578-c86a6338328a.png)

Genre: Rock

![image](https://user-images.githubusercontent.com/52060846/131236602-c7c29d76-5287-43d0-8266-80d5312eb4c5.png) ![image](https://user-images.githubusercontent.com/52060846/131236606-58cbb21f-89cb-4dfb-a69c-839c2857fad0.png)

Genre: Hip hop

![image](https://user-images.githubusercontent.com/52060846/131236612-12a25b26-135c-46d7-9263-2d40ff5cb1a4.png) ![image](https://user-images.githubusercontent.com/52060846/131236613-5723ebb0-3874-47b4-a376-c5669d7b5e75.png)

Full presentation: https://devpost.com/software/album-generator?ref_content=user-portfolio&ref_feature=in_progress

