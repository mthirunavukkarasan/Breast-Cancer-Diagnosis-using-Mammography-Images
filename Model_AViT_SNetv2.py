import torch
import tensorflow as tf
import torchvision.transforms as transforms
from keras.src.layers import DepthwiseConv2D
from vit_pytorch import ViT
import numpy as np
from Evaluation_nrml import evaluation
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, \
    BatchNormalization, concatenate, AveragePooling2D

def channel_shuffle(x, groups):
    _, w, h, c = x.get_shape().as_list()
    channels_per_group = c // groups

    x = tf.reshape(x, [-1, w, h, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, w, h, c])
    return x
def shuffle_unit(inputs, in_channels, out_channels, groups):
    shortcut = inputs

    # Grouped convolution
    branch_filters = out_channels // 4
    x = Conv2D(branch_filters, (1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = channel_shuffle(x, groups)
    x = DepthwiseConv2D((3, 3), padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)

    # Pointwise convolution
    x = Conv2D(out_channels - in_channels, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Concatenate with shortcut
    if out_channels == in_channels:
        x += shortcut
    else:
        x = tf.concat([x, shortcut], axis=-1)

    # Final channel shuffle
    x = channel_shuffle(x, groups)
    return x
def create_shufflenet(sol,input_shape, num_classes, groups=3):
    inputs = Input(input_shape)
    # Initial convolutional layer
    x = Conv2D(sol[2], (3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # ShuffleNet stages
    x = shuffle_stage(x, 24, 144, groups=groups, repeat=3)
    x = shuffle_stage(x, 144, 288, groups=groups, repeat=7)
    x = shuffle_stage(x, 288, 576, groups=groups, repeat=3)

    # Global average pooling and classifier
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='shufflenet',epoch=sol[3])
    return model


def shuffle_stage(inputs, in_channels, out_channels, groups, repeat):
    x = shuffle_unit(inputs, in_channels, out_channels, groups)
    for _ in range(repeat - 1):
        x = shuffle_unit(x, out_channels, out_channels, groups)
    return x

def conv_layer(conv_x, filters):
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(conv_x)
    conv_x = Dropout(0.2)(conv_x)

    return conv_x

def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate

    return block_x, filters


def transition_block(trans_x, tran_filters):
    trans_x = BatchNormalization()(trans_x)
    trans_x = Activation('relu')(trans_x)
    trans_x = Conv2D(tran_filters, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)

    return trans_x, tran_filters

def Model_AViT_SNetv2(image,Target,sol = None):
    if sol is None:
        sol = [5,5,5,5]
    # Load pre-trained Vision Transformer model
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=Target.shape[1],
        dim=768,
        depth=12,
        heads=sol[0],
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1

    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    # Extract features using the Vision Transformer model
    with torch.no_grad():
        features = model(input_tensor,epoch=sol[1])
    Feat = np.asarray(features)
    per = round(len(Feat) * 0.75)
    train_data = features[:per, :]
    train_target = Target[:per, :]
    test_data = features[per:, :]
    test_target = Target[per:, :]
    input_shape = (224, 224, 3)
    num_classes = train_target.shape[1]
    shufflenet_model = create_shufflenet(sol,input_shape, num_classes)
    shufflenet_model.compile(loss='binary_crossentropy', metrics=['acc'])
    shufflenet_model.fit(train_data, train_target)
    pred = np.round(shufflenet_model.predict(test_data)).astype('int')
    Eval = evaluation(pred, test_target)
    return Eval
