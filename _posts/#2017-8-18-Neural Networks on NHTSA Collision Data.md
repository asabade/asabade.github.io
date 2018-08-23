---
layout: post
title: Nueral Networks in TensorFlow on NHTSA collision data 
subtitle: Introduction to TensorFlow as a classification method with Neural Networks using NHTSA Collision Data with DUI Factors
bigimg: /img/path.jpg
---

**Addressing DUI Nationally with Open Data and Neural Networks**

Neural networks are the recent hot technique in machine learning. They have been used 
for image recognition, natural language processing, and music generation. among many 
other things. Their results often seem magical, but their operation is relatively straight- 
forward. I have started with a simple linear model known as logistic regression. 
I will demonstrate the limitations of the linear model and show how the neural networks 
overcome them. 

**Machine Learning Basics** 

Neural networks are an example of a $$supervised machine learning algorithm$$. In supervised machine leaming, we deal with 
two data structures, a feature matrix Xji and a label vector yj. Each column i of the feature matrix represents one particular 
feature. Each row j ff the feature matrix represents a particular observation. Each observation has a Single value for each 
feature, recorded as  Xji. Each observation also has an associated label y. 

![feature label](https://github.com/asabade/asabade.github.io/blob/master/_posts/1.PNG)

The goal of supervised machine learning is to build a function $$f$$ that can estimate the value of the label from the associated row 
of the feature matrix. 


Once we have such a function, we can predict the label for a new observation i withf(i). 
In this notebook, we will be considering only the case of binary classification, in which the labels .yj may take on only the 
values 0 and l. 
