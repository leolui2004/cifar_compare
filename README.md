# Explore Methods to Improve Accuracy on CIFAR-10 from 9% to 73%
My seventh project on github. Comparing accuracy between different methods and hyper-parameters on training CIFAR-10.

Feel free to provide comments, I just started learning Python for 3 monnths and I am now concentrating on data anylysis and web presentation.

## Reason To Compare
There are many ways to train a model for image recognition, and even for a simple model, there are many hyper-parameters which changing any one of them for a little bit can also cause a significant change in accuracy. By comparing different methods and hyper-parameters using CIFAR-10, lets see how accuracy varies from 9% to 70%.

## Methodology
Download CIFAR-10 images dataset and train the dataset using tensorflow. Images are pre-processed using OpenCV and labels are all one-hot encoded. Although many methods and hyper-parameters are being explored through the project, some settings are the same, for example categorical_crossentropy is being used for loss compile and Adam is being used for optimizer.

## Result
All the accuracy stated below are accuracy when evaluating test dataset.

### Round 1 - 10 epoch - Lets have a brief test
1. Test environment: Simple Dense, Color
> Epoch 10/10 10000/10000 - 1s 114us/sample - loss: 2.2962 - accuracy: 0.0988
>> Test Accuracy:  0.0926

The first try using purely 3 Dense layers with 1024, 256, 64 units. A plain list of data was sent to the network. As expectation, the result is very poor.

2. Test environment: Simple CNN, Color
> Epoch 10/10 10000/10000 - 18s 2ms/sample - loss: 1.1045 - accuracy: 0.6123
>> Test Accuracy:  0.4643

The second try using 3 Conv2D layers with 128(8x8), 256(4x4), 512(2x) units with Maxpooling 4x4, 2x2, 1x1 respectively, the second and the third one also connected to a Dropout(0.2) layer, which then Flatten and send to 2 Dense layers with 256, 64 units.

3. Test environment: Simple CNN, Grayscale
> Epoch 10/10 10000/10000 - 19s 2ms/sample - loss: 1.3606 - accuracy: 0.5090
>> Test Accuracy:  0.4712

For the third try, the network structure is same but images are converted to grayscale before training, thus the shape is 32x32x1 rather than 32x32x3.

To conclude on here, certainly an accuracy below 50% is far not enough, but here a simple result is, CNN is necessary.

### Round 2 - 50 epoch - Put all together, choose some to go further
1. Test environment: Simple CNN, Color, 5 batches combined to 1 batch, 10% validation
> Epoch 50/50 45000/45000 - 4s 99us/sample - loss: 0.0888 - accuracy: 0.9701
>> Test Accuracy:  0.4946 (overfit after 40epoch)

2. Test environment: Resnet, Color, 5 batches combined to 1 batch, 10% validation
> Epoch 50/50 45000/45000 - 60s 1ms/sample - loss: 0.2637 - accuracy: 0.9989
>> Test Accuracy:  0.6289 (overfit after 35epoch)

3. Test environment: Simple CNN, Grayscale, 5 batches combined to 1 batch, 10% validation
> Epoch 50/50 45000/45000 - 4s 91us/sample - loss: 0.0882 - accuracy: 0.9705
>> Test Accuracy:  0.5811 (overfit after 35epoch)

4. Test environment: Resnet, Grayscale, 5 batches combined to 1 batch, 10% validation
> Epoch 50/50 45000/45000 - 61s 1ms/sample - loss: 0.2586 - accuracy: 0.9999
>> Test Accuracy:  0.6074 (overfit after 40ep)

Resnet-56 was added to the test and pushed the record high to 62.89%, performances on Resnet-56 also surpasses all the control group. There also groups of color and grayscale, but the difference is not significant, so we will keep both for further tests.

### Round 3 - 100 epoch - Overcoming overfit

1. Test environment: Resnet, Color, 5 batches combined to 1 batch, 10% validation, Exponential decresing learning rate
> Epoch 100/100 45000/45000 - 49s 1ms/sample - loss: 0.1144 - accuracy: 1.0000 - val_loss: 2.0911 - val_accuracy: 0.7306 - lr: 1.3639e-06
>> Test Accuracy:  0.7081

2. 1. Test environment: Resnet, Grayscale, 5 batches combined to 1 batch, 10% validation, Exponential decresing learning rate
> Epoch 100/100 45000/45000 - 63s 1ms/sample - loss: 0.1461 - accuracy: 0.9999 - val_loss: 2.4751 - val_accuracy: 0.6620 - lr: 1.3639e-06
>> Test Accuracy:  0.6432

As we found significant over-fitting on last round, exponential decresing learning rate was added and further pushing the record high to 70.81%. The difference again between color and grayscale may not be big enough for conclusion, but because of that I just pick the color one and try to push to the limit.

### Round 4 - 80 epoch - Final way to 73.69% (Sample code uploaded)

The final try applied Amsgrad which is an advanced version of Adam, Data Augmentation using ImageDataGenerator provided by Keras. I think the result is already good enough because it only uses Resnet-56 which is not a SOTA architecture, neither using transfer learning nor extra training data from outside.

![image](https://github.com/leolui2004/cifar_compare/blob/master/result.png)

## Extra Round - Possible to get 90% easily?
This project was written on July 2020 and the SOTA 99%, which is not a surprising thing. However if you search on the net, people are claiming to get a result >90% easily with less than hundred of lines of code, but when you read the code they all used one loss function called  binary_crossentropy during compilation.

Lets demonstrate and see the result, I just randomly pick one of the test script on the above methods and change the loss function to binary_crossentropy.

> Epoch 1/10
> 45000/45000 - 13s - loss: 2.7665 - accuracy: 0.8200 - val_loss: 2.7680 - val_accuracy: 0.8195

The loss is large, just as large as using other methods to train the dataset, but it got 81.95% even for the 1st epoch, so definitely there is something wrong. And when I searched on the net, I found that this is because the way Keras used binary_accuracy by default if the loss function is binary_crossentropy, and it is not suitable for a model classifying 10 categories, so the accuracy is not true.

## What we can do to keep improving
The leaderboard shows different models achiving result mostly >95%. Those are not easy to implement and may take few times longer to train, but if we want to get a better result, keep learning new things is the only way.

Reference:
Leaderboard for CIFAR-10
https://paperswithcode.com/sota/image-classification-on-cifar-10
Sample code for building Resnet-56
https://qiita.com/shoji9x9/items/d09092b88ea42bd36a7b
Something about binary crossentropy and its accuracy
https://stackoverflow.com/questions/41327601/why-is-binary-crossentropy-more-accurate-than-categorical-crossentropy-for-multi
