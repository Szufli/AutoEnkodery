from keras.datasets import mnist
from keras import models
from keras.models import Model
from keras.models import model_from_json
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from numpy import expand_dims
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Activation

import warnings; warnings.filterwarnings('ignore')

(train, _), (test, _) = mnist.load_data()

train = train.astype('float32') / 255.
test = test.astype('float32') / 255.
train = train.reshape(-1, 28, 28, 1)
test = test.reshape(-1, 28, 28, 1)
noise_factor = 0.5
noiseT = noise_factor * np.random.normal(loc= 0.5, scale= 0.5, size= train.shape)
noiseZ = noise_factor * np.random.normal(loc= 0.5, scale= 0.5, size= test.shape)
noisedTrain = train + noiseT
noisedTest = test + noiseZ
noisedTrain = np.clip(noisedTrain, 0., 1.)
noisedTest = np.clip(noisedTest, 0., 1.)

# Enkoder
Wejście = Input(shape=(28,28,1), name='Enkoder_In')
Enkoder_Konwulacyjna_01 = Conv2D(32, (3, 3), padding='same',kernel_initializer='he_uniform', activation='relu', name='Enkoder_KonwW1')(Wejście)
Enkoder_Pool_01 = MaxPooling2D(pool_size=(2,2), padding='same', name='Enkoder_PoolW1')(Enkoder_Konwulacyjna_01)
Enkoder_Konwulacyjna_02 = Conv2D(32,(3, 3), padding='same',kernel_initializer='he_uniform', activation='relu', name='Enkoder_KonwW2')(Enkoder_Pool_01)
Enkoder_Pool_02 = MaxPooling2D(pool_size=(2,2), padding='same', name='Enkoder_PoolW2')(Enkoder_Konwulacyjna_02)

#Wąskie gardło
BottleNeck = Conv2D(1, (3, 3), padding='same', activation='sigmoid', name='BootleNeck')(Enkoder_Pool_02)


# Instantiate the Encoder Model
encoder = Model(inputs = Wejście, outputs = BottleNeck, name="Enkoder")
encoder.summary()

# Decoder
Dekoder_Konwulacyjna_01 = Conv2D(32, (3, 3), padding='same',kernel_initializer='he_uniform', activation='relu', name ='Dekoder_KonwW1')(BottleNeck)
Dekoder_Up_01 = UpSampling2D((2,2),interpolation="bilinear",name ='Dekoder_UpSampW1')(Dekoder_Konwulacyjna_01)
Dekoder_Konwulacyjna_02 = Conv2D(32, (3, 3), padding='same',kernel_initializer='he_uniform', activation='relu', name ='Dekoder_KonwW2')(Dekoder_Up_01)
Dekoder_Up_02 = UpSampling2D((2,2),interpolation="bilinear",name ='Dekoder_UpSampW2')(Dekoder_Konwulacyjna_02)
Dekoder_Konwulacyjna_03 = Conv2D(16, (3, 3), padding='same',kernel_initializer='he_uniform', activation='relu', name ='Dekoder_KonwW3')(Dekoder_Up_02)
#Wyjście
Wyjście = Conv2D(1,(3, 3), padding='same', activation='sigmoid', name ='Dekoder_Out')(Dekoder_Konwulacyjna_03)


autoenkoder = Model(inputs=Wejście, outputs=Wyjście, name="Autoenkoder")
autoenkoder.compile(loss='mse', optimizer='adam')
autoenkoder.fit(noisedTrain, train,
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_split=1/6)


tr_loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(tr_loss)+1)

fig = plt.figure(figsize=(8, 4))
fig.tight_layout()
plt.plot(epochs, tr_loss,'r')
plt.plot(epochs, val_loss,'b')
plt.title('Funkcja straty')
plt.ylabel('Wartość błędu średniokwadratowego')
plt.xlabel('Ilość epoch')
plt.legend(['Trening', 'Walidacja'], loc='upper right')
plt.show()
# Save the Encoder
model_json = encoder.to_json()
with open("/usr/autoencoder/logs/Encoder_model.json", "w") as json_file:
    json_file.write(model_json)
encoder.save_weights("/usr/autoencoder/logs/Encoder_weights.h5")

# Save the Autoencoder
model_json = autoencoder.to_json()
with open("/usr/autoencoder/logs/Autoencoder_model.json", "w") as json_file:
    json_file.write(model_json)
autoencoder.save_weights("/usr/autoencoder/logs/Autoencoder_weights.h5")
