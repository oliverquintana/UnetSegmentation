from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal

def unet():

    input_img = Input(shape=(256, 256, 1))

    # Encoder
    # Step 1
    conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = he_normal())(input_img)
    conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = he_normal())(conv1)
    drop1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size = (2,2))(drop1)
    # Step 2
    conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = he_normal())(pool1)
    conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = he_normal())(conv2)
    drop2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size = (2,2))(drop2)
    # Step 3
    conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = he_normal())(pool2)
    conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = he_normal())(conv3)
    drop3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D(pool_size = (2,2))(drop3)
    # Step 4
    conv4 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = he_normal())(pool3)
    conv4 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = he_normal())(conv4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size = (2,2))(conv4)
    # Step 5
    conv5 = Conv2D(1024, (3,3), activation = 'relu', padding = 'same', kernel_initializer = he_normal())(pool4)
    conv5 = Conv2D(1024, (3,3), activation = 'relu', padding = 'same', kernel_initializer = he_normal())(conv5)
    drop5 = Dropout(0.2)(conv5)
    #pool5 = MaxPooling2D(pool_size = (2,2))(conv5)

    # Decoder
    # Step 1
    up1 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = he_normal())(UpSampling2D(size = (2,2))(drop5))
    conc1 = concatenate([conv4, up1], axis = 3)
    conv6 = Conv2D(512, (3,3), activation = 'relu', padding = "same", kernel_initializer = he_normal())(conc1)
    conv6 = Conv2D(512, (3,3), activation = "relu", padding = "same", kernel_initializer = he_normal())(conv6)
    drop6 = Dropout(0.2)(conv6)
    # Step 2
    up2 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = he_normal())(UpSampling2D(size = (2,2))(drop6))
    conc2 = concatenate([conv3, up2], axis = 3)
    conv7 = Conv2D(256, (3,3), activation = 'relu', padding = "same", kernel_initializer = he_normal())(conc2)
    conv7 = Conv2D(256, (3,3), activation = "relu", padding = "same", kernel_initializer = he_normal())(conv7)
    drop7 = Dropout(0.2)(conv7)
    # Step 3
    up3 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = he_normal())(UpSampling2D(size = (2,2))(drop7))
    conc3 = concatenate([conv2, up3], axis = 3)
    conv8 = Conv2D(128, (3,3), activation = 'relu', padding = "same", kernel_initializer = he_normal())(conc3)
    conv8 = Conv2D(128, (3,3), activation = "relu", padding = "same", kernel_initializer = he_normal())(conv8)
    drop8 = Dropout(0.2)(conv8)
    # Step 4
    up4 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = he_normal())(UpSampling2D(size = (2,2))(drop8))
    conc4 = concatenate([conv1, up4], axis = 3)
    conv9 = Conv2D(64, (3,3), activation = 'relu', padding = "same", kernel_initializer = he_normal())(conc4)
    conv9 = Conv2D(64, (3,3), activation = "relu", padding = "same", kernel_initializer = he_normal())(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input_img, conv10)

    return model
