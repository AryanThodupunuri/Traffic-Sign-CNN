# Convolutional Neural Network to Classify Traffic Sign Images

This project aims to train a LeNet-5 Convolutional Neural Network to classify user-provided traffic signs. The model is trained and tested on approximately 40,000 real photos encompassing 43 types of German traffic signs, sourced from The German Traffic Sign Recognition Benchmark (GTSRB): (https://benchmark.ini.rub.de/gtsrb_dataset.html). Overall, this project was great because it taught me how to use a convolutional neural network to classify any images. I need to learn the process for understanding how to choose the optimal number of hidden layers and nodes. Maybe later on, I can experiment by adding different hyperparameters like learning rate, batch size, and optimizer settings to optimize model performance. I also wonder what the model would have looked like if I used pre-trained models such as ResNET, VGG, or MobileNET as feature extractors and then fine-tuning them for classification.

Included Files:

- traffic_signs_classification_lenet5.ipynb: Jupyter Notebook containing the code for building and training the model.
- keras_model: Folder containing the saved model.
- streamlit_app.py: Streamlit application for interactive testing of the trained model.
Notebook with built and trained model: **traffic_signs_classification_lenet5.ipynb**

Article: https://towardsdatascience.com/classification-of-traffic-signs-with-lenet-5-cnn-cb861289bd62

Link to Streamlit app: https://share.streamlit.io/andriigoz/traffic_signs_classification
