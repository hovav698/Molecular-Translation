**Dataset description**

The dataset contains more then 2 million images of chemicals, with the objective of predicting the corresponding International Chemical Identifier (InChI) text string of the image.
Here is an example from the dataset of random image:

![image](https://user-images.githubusercontent.com/71300410/121773308-d040e400-cb83-11eb-9d4f-ed08e4031a8a.png)

and it's corresponding International Chemical Identifier:

InChI='1S/C15H21N3O5/c1-2-7-18-12(16)11(13(20)17-15(18)22)10(19)8-23-14(21)9-5-3-4-6-9/h9H,2-8,16H2,1H3,(H,17,20,22)'


**Project Goal**

The goal of the project is to create a model that learns to decode and translate the input image to the corresponding Chemical Identifier text. This will be done using the transformer model.

**How the model works?**

The transformer model was originally developed for the purpose of solving NLP task like text translation. This project is very similar to the text translation task, the difference is that the input is an image instead of text sequence. So the way to do it is pretty simple, we need to extract the image features and convert it to a sequence in a shape that will fit to the transformer encoder module. We will do it using a Resnet18 convolution model, the image feautre will be extracted, reshaped and fed into the  encoder module. 
The encoder side will encode the image sequence, while the decoder side will receive the encoded sequence, and together with its own predicted tokens information will predict the next token, until the "end" token will be predicted. It will work exactly like described in "Attention Is All You Need" paper. 
  
  
**Data Preperation**
  
 We will need to create a tokenizer for the Chemical Identifier text sequence. We wil do it by split the original string to its componets, each number and letter will be convert to it's own token, and we also need to consider some special tokens, for example the one that start with "/". We will then tokenize all the text in the data, and will add padding according to the maximum text length.
We will create data generator, it will generate batchs of images and it corresponding chemical string, and feed into the model.
  
  
 **Dataset size consideration**
  
  The dataset contains more then 2 millions images, it will take us too long to train the model on all the data. Therefore I chose to train the model on a random subset of 10,000 images and text sequence. 
  
  
**Result**
  
 In the training steps I was using teacher forcing method - I gave the encoder model the correct previews token, because it makes the model learn better the sequence pattern, much better then useing the predicted token. With the teacher forcing I was able to reach around 60% accuracy in the test set and 90% accuracy in the train set.
However when it came to the prediction part, I was no longer used the teacher forcing method, and the prediction result were bad with very low accuracy. 
I think that if I will train the model on much larger data, the model can generalized better and I would get better result. 
Another change that I will try in the future is using the visual transformer model. It is similar to what I did, the difference is that instead of using convolution model for extracting the image feautres, I will split the image into patches of smaller images, and add positional encoding to every patch, then use it as sequences. I think this way can help to improve the model result significantly. 
Will be updated.
  

