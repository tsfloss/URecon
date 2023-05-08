import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
from tensorflow import pad
    
class PeriodicPadding3D(Layer):
    def __init__(self, **kwargs):
        super(PeriodicPadding3D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.grid = input_shape[1]
        self.indices = np.append(np.insert(np.arange(self.grid),0,self.grid-1),0).astype(np.int32)
    
    def call(self, x):
        x = tf.gather(x,self.indices,axis=1)
        x = tf.gather(x,self.indices,axis=2)
        x = tf.gather(x,self.indices,axis=3)
        return x

def create_localization_module(input_layer, current_grid, n_filters):
    layer1 = PeriodicPadding3D()(input_layer)
    convolution1 = create_convolution_block(layer1, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, current_grid, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    layer1 = PeriodicPadding3D()(up_sample)
    convolution = create_convolution_block(layer1, n_filters)
    return convolution


def create_context_module(input_layer, current_grid,  n_level_filters, dropout_rate=0.2, data_format="channels_last"):
    layer1 = PeriodicPadding3D()(input_layer)
    convolution1 = create_convolution_block(input_layer=layer1, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    layer2 = PeriodicPadding3D()(dropout)
    convolution2 = create_convolution_block(input_layer=layer2, n_filters=n_level_filters)
    return convolution2


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=LeakyReLU,
                             padding='valid', strides=(1, 1, 1), instance_normalization=False):
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    layer = InstanceNormalization()(layer)
    return activation()(layer)

class displacement_tensors(tf.keras.layers.Layer):
    def __init__(self,BoxSize):
        super(displacement_tensors, self).__init__()
        self.BoxSize = BoxSize
        
    def build(self, input_shape):
        self.grid = input_shape[1]
        self.k = tf.meshgrid(
        2 * np.pi * np.fft.fftfreq(self.grid, self.BoxSize/self.grid),
        2 * np.pi * np.fft.fftfreq(self.grid, self.BoxSize/self.grid),
        2 * np.pi * np.fft.rfftfreq(self.grid, self.BoxSize/self.grid),  # Note rfft.
        indexing="ij")
        self.kx = tf.cast(self.k[0],dtype=tf.complex64)
        self.ky = tf.cast(self.k[1],dtype=tf.complex64)
        self.kz = tf.cast(self.k[2],dtype=tf.complex64)
        self.knorm2 = self.kx**2 + self.ky**2 + self.kz**2
        self.knorm2 = tf.where(self.knorm2 != 0., self.knorm2, 1.+0.j)

    def call(self, inputs):
        inputs = tf.einsum('bhwdc->bchwd',inputs)
        inputs_fft = tf.signal.rfft3d(inputs)
        
        inputs_fft *= 1j/self.knorm2
        psix = inputs_fft * self.kx
        psiy = inputs_fft * self.ky
        psiz = inputs_fft * self.kz
        psixy= psix * self.ky*1j
        psixz= psix * self.kz*1j
        psiyz= psiy * self.kz*1j
        
        outputs_fft = tf.concat([psix,psiy,psiz,psixy,psixz,psiyz],axis=1)
        
        output = tf.signal.irfft3d(outputs_fft)
        output = tf.einsum('bchwd->bhwdc',output)
        return output

def UNET3D(image_size, n_base_filters=16, depth=5, dropout_rate=0.3, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=tf.keras.losses.mse):

    inputs = Input((image_size, image_size, image_size, 1), name='density')
    y = inputs
    x = displacement_tensors(BoxSize)(y)
    concat = tf.keras.layers.Concatenate()([y,x])
    x = concat

    level_output_layers = list()
    level_filters = list()
    
    current_grid = image_size
    
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        # n_level_filters = n_base_filters / (2**level_number)
        # n_level_filters = n_base_filters
        level_filters.append(n_level_filters)

        if x is concat:
            x = PeriodicPadding3D()(x)
            x = create_convolution_block(x, n_level_filters)
        else:
            x = PeriodicPadding3D()(x)
            x = create_convolution_block(x, n_level_filters, strides=(2, 2, 2))
            current_grid//=2
        
        previous_block = x
        x = create_context_module(x, current_grid, n_level_filters, dropout_rate=dropout_rate)
        x = Add()([previous_block, x])
        level_output_layers.append(x)

    for level_number in range(depth - 2, -1, -1):
        current_grid*=2
        x = create_up_sampling_module(x, current_grid, level_filters[level_number])
        x = Concatenate()([level_output_layers[level_number], x])
        x = create_localization_module(x, current_grid, level_filters[level_number])

    x = Conv3D(1,kernel_size=(1,1,1),strides=(1,1,1))(x)
    outputs = x
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer(learning_rate=initial_learning_rate), loss=loss_function)
    return model