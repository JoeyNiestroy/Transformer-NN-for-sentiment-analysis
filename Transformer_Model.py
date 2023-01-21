import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import datetime

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

"""Loading in Training Data"""
df = pd.read_csv("Training_Data.csv")

"""Formatting and adding padding"""
x_train = []
for sent in df["Encoded"]:
    temp = eval(sent)
    if len(temp) < 35:
        temp.extend([0]*(35-len(temp)))
    x_train.append(np.array(temp))
x_train = np.asarray(x_train)

y_train = []
for sent in df["Sentiment"]:
    if sent == 0:
        y_train.append(0)
    else:
        y_train.append(1)
y_train = np.asarray(y_train)

"""10K Samples taken as validation data"""
x_valid = x_train[240000:]
y_valid = y_train[240000:]

x_train = x_train[:240000]
y_train = y_train[:240000]

""""Declaration of model paramters"""

vocab_size = 46971
maxlen = 35

embed_dim = 126  
num_heads = 2  
ff_dim = 32  


"""Designing and training model"""
inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)
log_dir = "logs/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=3, update_freq='epoch')

# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath =  "saved-model-{epoch:02d}.hdf5",
#     save_weights_only=False,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=False)

model.fit(x=x_train, 
          y=y_train, 
          epochs=5, 
          validation_data=(x_valid, y_valid), 
          callbacks=[tensorboard_callback])
model.save("Transformer_Model")