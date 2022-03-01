import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.linalg import einsum
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import backend as K

class additiveAttention(layers.Layer):
    # addictive word level attention
    def __init__(self, output_dim, seed=42, **kwargs):
        self.output_dim = output_dim
        self.seed = seed
        super(additiveAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # [batch, title_size, pre_output_dim]
        self.W = self.add_weight(
            name='weight',
            shape=(int(input_shape[-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True
        )
        self.b = self.add_weight(
            name='bias',
            shape=(self.output_dim, ),
            initializer=keras.initializers.Zeros(),
            trainable=True
        )
        self.q = self.add_weight(
            name='query',
            shape=(self.output_dim, 1),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True
        )
        super(additiveAttention, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, inputs, mask=None, **kwargs):
        '''
        q*tanh(wx+b)
        softmax
        continugous sum
        '''
        x = K.tanh(K.dot(inputs, self.W) + self.b)
        x = K.dot(x, self.q)
        # ?
        attention = K.squeeze(x, axis=2)

        if mask is None:
            attention = K.exp(attention)
        else:
            attention = K.exp(attention) * K.cast(mask, dtype='float32')

        attention_weight = attention / (K.sum(attention, axis=-1, keepdims=True) + K.epsilon()) # noise?
        attention_weight = K.expand_dims(attention_weight)

        weighted_summation = K.sum(inputs * attention_weight, axis = 1)
        return weighted_summation

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class multiHeadSelfAttention(layers.Layer):
    '''
    word-level, multi-head, self-attention
    '''
    def __init__(self, head_num, head_dim, seed=0, mask_right=False, **kwargs):

        self.head_num = head_num
        self.head_dim = head_dim
        self.output_dim = head_num * head_dim
        self.mask_right = mask_right
        self.seed = seed
        super(multiHeadSelfAttention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    # initialize weight before call()
    def build(self, input_shape):

        w_init = keras.initializers.glorot_uniform(seed=self.seed)

        self.q_w = self.add_weight(
            name='q_seq_weight',
            shape=(int(input_shape[0][-1]), self.output_dim),
            initializer=w_init,
            trainable=True
        )
        self.k_w = self.add_weight(
            name='k_seq_weight',
            shape=(int(input_shape[1][-1]), self.output_dim),
            initializer=w_init,
            trainable=True
        )
        self.v_w = self.add_weight(
            name='v_seq_weight',
            shape=(int(input_shape[2][-1]), self.output_dim),
            initializer=w_init,
            trainable=True
        )
        super(multiHeadSelfAttention, self).build(input_shape)


    def call(self, param_lst):
        '''
        Scaled dot-Product Attention
        Attention(Q, K, V)=softmax(QK/√dk)V
        '''
        # embedded news title
        x1, x2, x3 = param_lst

        q_seq = K.dot(x1, self.q_w)
        k_seq = K.dot(x2, self.k_w)
        v_seq = K.dot(x3, self.v_w)

        q_len = None
        v_len = None

        # output_dim splits, permute axes
        # shape = [batch, head_num, title_size, head_output_dim]
        q_seq = K.reshape(q_seq, shape=(-1, K.shape(q_seq)[1], self.head_num, self.head_dim))
        q_seq = K.permute_dimensions(q_seq, pattern=(0, 2, 1, 3))

        k_seq = K.reshape(k_seq, shape=(-1, K.shape(k_seq)[1], self.head_num, self.head_dim))
        k_seq = K.permute_dimensions(k_seq, pattern=(0, 2, 1, 3))

        v_seq = K.reshape(v_seq, shape=(-1, K.shape(v_seq)[1], self.head_num, self.head_dim))
        v_seq = K.permute_dimensions(v_seq, pattern=(0, 2, 1, 3))

        # einsum rule: abij, abkj -> abik axis 2, 3,  batch_size a, b heads, i q_seq, k k_seq
        # compute dot products of the query with all k_seq, divide √dk
        y = einsum('abij, abkj -> abik', q_seq, k_seq) / K.sqrt(K.cast(self.head_dim, dtype='float32'))
        y = K.permute_dimensions(y, pattern=(0, 3, 2, 1))

        y = self.Mask(y, v_len, 'add')
        # shape = [batch, head_dim, title_size, head_num]
        y = K.permute_dimensions(y, pattern=(0, 3, 2, 1))

        # if self.mask_right:
        #     ones = K.ones_like(y[:1, :1])
        #     lower_triangular = K.tf.matrix_band_part(ones, num_lower=-1, num_upper=0)
        #     mask = (ones - lower_triangular) * 1e12
        #     y = y - mask

        y = K.softmax(y)

        O_seq = einsum('abij, abjk -> abik', y, v_seq)
        O_seq = K.permute_dimensions(O_seq, pattern=(0, 2, 1, 3))

        O_seq = K.reshape(O_seq, shape=(-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, q_len, 'mul')

        return O_seq

    def Mask(self, inputs, seq_len, mode='add'):
        # title sequence auto complete positions make no sense in attention
        if seq_len is None:
            return inputs
        else:
            #Computes the one - hot representation
            mask = K.one_hot(indices=seq_len[:, 0], num_classes=K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, axis=1)

            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)

            if mode == 'mul':
                return inputs * mask
            elif mode == 'add':
                return inputs - (1 - mask) * 1e12

    def get_config(self):
        layer_config = super(multiHeadSelfAttention, self).get_config()
        layer_config.update(
            {
                "head_num": self.head_num,
                "head_dim": self.head_dim,
                "mask_right": self.mask_right,
            }
        )
        return layer_config

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)
