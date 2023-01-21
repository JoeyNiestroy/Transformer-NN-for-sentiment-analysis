# Transformer-NN-for-sentiment-analysis
A follow-up to previous twitter sentiment project using transformer based network for sentiment analysis

Training and testing data was used from previous twitter sentiment repo, data was simply index encoded tweets. Total training data was 250k samples(File to large to upload, but shoot me an email if you'd like the data). Keras was used to build a transformer based NN, only a single transformer was used. 10K training samples were used as validation data, optimizer was adam with a lr = 5e-5, and the model was trained for 3 epochs. (Transformer_Model.py)  The following graph is from Tensorboard and the training logs.
![my_graph](https://user-images.githubusercontent.com/106636917/213883290-a331958d-193e-457d-a6c5-127de5c61c7a.JPG)
Peak validation accuracy ocurrs at epoch 2, so that model was tested against clean test data of around 25k (Test_data.csv). An accuracy of 79.4% was achieved. Compared to the 73% accuracy of the previous logistic regression approach this is an almost 8% improvement.
