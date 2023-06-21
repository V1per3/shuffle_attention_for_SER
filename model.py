import tensorflow as tf
import keras.layers as nn
from keras import Model



def channel_shuffle(input, groups):
    batchsize, num_channels, height, width = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]]
    channels_per_group = num_channels // groups
    # grouping, 通道分组
    output = tf.reshape(input,(batchsize, groups, channels_per_group, height, width))
    # channel shuffle, 通道洗牌
    output = tf.transpose(output, (0,2,1,3,4))

    output = tf.reshape(output,(batchsize, -1, height, width))

    return output

class SACNN(Model):
    def __init__(self,groups = 16,channel = 128, out_size=4):
        super(AACNN, self).__init__()
        self.groups = groups
        self.channel = channel
        self.cweight = tf.Variable(tf.zeros([1, channel // (2 * groups), 1, 1]))
        self.cbias = tf.Variable(tf.ones([1, channel // (2 * groups), 1, 1]))
        self.sweight = tf.Variable(tf.zeros([1, channel // (2 * groups), 1, 1]))
        self.sbias = tf.Variable(tf.ones([1, channel // (2 * groups), 1, 1]))
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.gn = tf.keras.layers.GroupNormalization(groups=channel // (2 * groups), axis=-1)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

        self.conv1 = nn.Conv2D(32, (3, 3), padding='same', data_format='channels_last', )

        self.conv1a = nn.Conv2D(16, (10, 2), padding='same', data_format='channels_last', )
        self.conv1b = nn.Conv2D(16, (2, 8), padding='same', data_format='channels_last', )
        self.conv2 = nn.Conv2D(32, (3, 3), padding='same', data_format='channels_last', )
        self.conv3 = nn.Conv2D(48, (3, 3), padding='same', data_format='channels_last', )
        self.conv4 = nn.Conv2D(64, (3, 3), padding='same', data_format='channels_last', )
        self.conv5 = nn.Conv2D(80, (3, 3), padding='same', data_format='channels_last', )
        self.conv6 = nn.Conv2D(128, (3, 3), padding='same', data_format='channels_last', )
        self.maxp = nn.MaxPool2D((2, 2))

        self.bn1a = nn.BatchNormalization(3)
        self.bn1b = nn.BatchNormalization(3)
        self.bn2 = nn.BatchNormalization(3)
        self.bn3 = nn.BatchNormalization(3)
        self.bn4 = nn.BatchNormalization(3)
        self.bn5 = nn.BatchNormalization(3)
        self.bn6 = nn.BatchNormalization(3)

        self.dropout = nn.Dropout(rate=0.5)
        # self.residual1 = nn.Conv2D(32, (1, 1), padding='same', data_format='channels_last')
        # self.residual2 = nn.Conv2D(48, (1, 1), padding='same', data_format='channels_last')
        # self.residual3 = nn.Conv2D(64, (1, 1), padding='same', data_format='channels_last')
        # self.residual4 = nn.Conv2D(80, (1, 1), padding='same', data_format='channels_last')
        # self.residual5 = nn.Conv2D(128, (1, 1), padding='same', data_format='channels_last')
        self.flatten = nn.Flatten(data_format='channels_last')
        self.fc = nn.Dense(out_size, activation='softmax')



    def call(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa = tf.nn.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = tf.nn.relu(xb)
        x = tf.concat([xa, xb], 1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.maxp(x)

        # residual = self.residual1(x)  # 添加残差连接
        # x = x + residual
        # x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)
        x = self.maxp(x)

        # residual = self.residual2(x)  # 添加残差连接
        # x = x + residual
        # x = self.dropout(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = tf.nn.relu(x)

        # residual = self.residual3(x)  # 添加残差连接
        # x = x + residual
        # x = self.dropout(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = tf.nn.relu(x)#(32, 64, 20, 80)

        # residual = self.residual4(x)  # 添加残差连接
        # x = x + residual
        # x = self.dropout(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = tf.nn.relu(x)

        # residual = self.residual5(x)  # 添加残差连接
        # x = x + residual

        # ShuffleAttention
        N,H,W,C = [x.shape[0],x.shape[1],x.shape[2],x.shape[3]]
        x = tf.reshape(x,(N*self.groups,-1,H,W))
        x_0,x_1 = tf.split(x,2,1)

        xn = self.avg_pool(x_0)
        xn = xn[:, tf.newaxis, tf.newaxis, :]
        xn = self.cweight * xn +self.cbias
        xn = x_0 * self.sigmoid(xn)

        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        x = tf.concat([xn,xs],1)
        x = tf.reshape(x,(N,-1,H,W))

        x = channel_shuffle(x,2)#(32, 80, 64, 20)
        # x = tf.reshape(x,(x.shape[0],x.shape[2],x.shape[3],x.shape[1]))

        x = tf.nn.relu(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x