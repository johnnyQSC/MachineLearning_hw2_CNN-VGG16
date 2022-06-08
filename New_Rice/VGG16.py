from keras import layers,models,optimizers
import keras
from keras.applications import vgg16
from tensorflow.python.keras.engine.training import Model
from keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras import optimizers
from keras.applications.vgg16 import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
#VGG16
# conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(504,378,3))
base_dir=(r'')
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'test')
conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(504,378,3))
model=models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))
model.summary()
conv_base.trainable=False
train_datagen=ImageDataGenerator(rescale=1./255)
validation_datagen=ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory(train_dir,target_size=(504,378),batch_size=50,class_mode='categorical')
validation_generator=validation_datagen.flow_from_directory(validation_dir,shuffle=False,target_size=(504,378),batch_size=20,class_mode='categorical')
model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(learning_rate=2e-5),metrics=['acc'])  #1e-4
history=model.fit_generator(train_generator,steps_per_epoch=train_generator.samples/train_generator.batch_size,epochs=20,validation_data=validation_generator,validation_steps=validation_generator.samples/validation_generator.batch_size)

validation_generator.reset()
import time
model_evaluate=model.evaluate_generator(validation_generator,workers=0)  #準確度1筆
start=time.time()
model_predict=model.predict_generator(validation_generator,verbose=1)#混淆矩陣可以跑出5筆準確度
end=time.time()
print(end-start)
print('ACC=',model_evaluate[1])
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
#print(confusion_matrix(y_true=y_true,y_pred= y_pred))
y_true=validation_generator.classes
y_pred = np.argmax(model_predict, axis=1)
confmat=confusion_matrix(y_true,y_pred)

fig, ax = plt.subplots(figsize=(5,5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
plt.xlabel('Predicted label')        
plt.ylabel('True label')
plt.show()
print(classification_report(y_true,y_pred))
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training acc')

plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training & Validation acc')
plt.legend()
plt.figure()
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training & Validation loss')
plt.show()