import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import numpy as np

# 데이터 경로 설정
data_dir = 'C:/Users/HUSTAR12/Desktop/Squat/Dataset'
tf.keras.backend.set_floatx('float64')
epochs = 10
drop = 0.5
img_size = (128,128)

#ReLU
model = Sequential([
    Conv2D(8, 5, activation = 'relu', input_shape = (img_size[0], img_size[1], 1)),
    MaxPool2D(3),
    Conv2D(16, 4, activation = 'relu'),
    MaxPool2D(2),
    Conv2D(32, 3, activation = 'relu'),
    Flatten(),
    Dense(32, activation = 'relu'),
    Dropout(drop),
    Dense(8, activation = 'relu'),
    Dense(3, activation = 'softmax')
])

# 옵티마이저 : adam
# 손실함수 : categorical_crossentropy

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# 모델 구조 확인
model.summary()

# 이미지 설정
datagen = ImageDataGenerator(
    rescale = 1. / 255.,
    shear_range = 0.2,
    zoom_range = 0.05,
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.05,
    brightness_range = [1, 1.5],
    horizontal_flip = True,
    dtype = tf.float64)
# train 이미지 color_mode : grayscale로 변경

train_generator = datagen.flow_from_directory(
    'C:/Users/HUSTAR12/Desktop/Squat/Dataset/Train_data',
    target_size = img_size,
    color_mode = 'grayscale',
    batch_size = 32,
    shuffle = True,
    class_mode='categorical')

test_datagen = ImageDataGenerator(
    rescale = 1. / 255.,
    dtype = tf.float64)

# test 이미지 color_mode : grayscale로 변경

test_generator = test_datagen.flow_from_directory(
    'C:/Users/HUSTAR12/Desktop/Squat/Dataset/Validation_data',
    target_size = img_size,
    color_mode = 'grayscale',
    batch_size = 16,
    shuffle = True,
    class_mode='categorical')

# 가중치 설정(사용안함)
# class_weights = class_weight.compute_class_weight(
#                    'balanced',
#                    np.unique(train_generator.classes), 
#                    train_generator.classes)


history = model.fit(train_generator, 
          validation_data = test_generator,
          epochs = epochs,
          shuffle = True,
          #class_weight = class_weights,
          workers = 8,
          max_queue_size = 512)

# 모델 저장
model.save('saved/saved11.h5')

# 정확도, 손실도 그래프 표시
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()