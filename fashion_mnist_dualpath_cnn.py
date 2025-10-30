"""
DualPath-AttentionNet - Arquitetura CNN Inédita para Fashion-MNIST
=====================================================================

DESAFIO 03e - AC3
Arquitetura criada especificamente para este desafio, sem replicar arquiteturas clássicas.

CONCEITO DA ARQUITETURA:
------------------------
A DualPath-AttentionNet é uma arquitetura inovadora que processa imagens através de
dois caminhos paralelos (dual-path) com diferentes escalas de receptive fields,
integrados através de mecanismos de atenção espacial.

CARACTERÍSTICAS ÚNICAS:
-----------------------
1. Processamento Dual-Path: Dois caminhos paralelos processam a imagem simultaneamente
   - Path 1 (Detail): Captura detalhes finos com filtros pequenos (3x3)
   - Path 2 (Context): Captura contexto amplo com filtros maiores (5x5) e dilatação

2. Mecanismo de Atenção Espacial: Camadas de atenção customizadas que aprendem
   quais regiões da imagem são mais importantes para a classificação

3. Fusão Adaptativa: Os dois caminhos são fundidos através de concatenação seguida
   de convolução 1x1 para combinar informações de diferentes escalas

4. Convoluções Dilatadas: Aumentam o receptive field sem aumentar parâmetros

5. Regularização Avançada: Batch Normalization, Spatial Dropout e L2 regularization

REFERÊNCIAS TEÓRICAS:
---------------------
- Multi-scale feature processing: Inspirado em conceitos de pirâmides de features
  (Lin et al., 2017 - Feature Pyramid Networks)
- Attention mechanisms: Baseado em trabalhos de attention em CNNs
  (Woo et al., 2018 - CBAM: Convolutional Block Attention Module)
- Dilated convolutions: Yu & Koltun, 2016 - Multi-Scale Context Aggregation
- Dual-path processing: Conceito inspirado em processamento visual humano
  (ventral/dorsal streams)

JUSTIFICATIVA PARA FASHION-MNIST:
----------------------------------
Fashion-MNIST apresenta desafios únicos:
- Imagens pequenas (28x28) exigem preservação de detalhes
- 10 classes com variações intra-classe significativas
- Alguns itens compartilham características (shirt vs t-shirt, sneaker vs ankle boot)

Nossa arquitetura aborda isso através de:
- Dual-path para capturar tanto detalhes quanto contexto global
- Atenção para focar em regiões discriminativas
- Múltiplas escalas de processamento
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime


# ============================================================================
# CONFIGURAÇÕES E HIPERPARÂMETROS
# ============================================================================

class Config:
    """
    Configurações e hiperparâmetros do modelo

    Justificativas:
    ---------------
    - IMG_SIZE: 28x28 do Fashion-MNIST
    - BATCH_SIZE: 64 - balanceamento entre estabilidade e velocidade
    - EPOCHS: 100 com early stopping para evitar overfitting
    - LEARNING_RATE: 0.001 - taxa padrão do Adam, comprovadamente eficaz
    - L2_REG: 1e-4 - regularização L2 para prevenir overfitting
    - DROPOUT_RATE: 0.3 - dropout moderado para regularização
    - SPATIAL_DROPOUT: 0.2 - dropout espacial para manter coerência espacial
    """
    IMG_SIZE = 28
    IMG_CHANNELS = 1
    NUM_CLASSES = 10

    # Hiperparâmetros de treinamento
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001

    # Regularização
    L2_REG = 1e-4
    DROPOUT_RATE = 0.3
    SPATIAL_DROPOUT = 0.2

    # Data augmentation
    ROTATION_RANGE = 10
    WIDTH_SHIFT = 0.1
    HEIGHT_SHIFT = 0.1
    ZOOM_RANGE = 0.1

    # Classes do Fashion-MNIST
    CLASS_NAMES = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]


# ============================================================================
# BLOCOS CUSTOMIZADOS DA ARQUITETURA
# ============================================================================

class SpatialAttention(layers.Layer):
    """
    Camada de Atenção Espacial Customizada

    DESCRIÇÃO:
    ----------
    Aprende a dar peso diferenciado para diferentes regiões espaciais da imagem.
    Computa um mapa de atenção através de operações de pooling e convolução.

    MATEMÁTICA:
    -----------
    1. Feature aggregation:
       f_avg = AvgPool(F) -> [H, W, 1]
       f_max = MaxPool(F) -> [H, W, 1]

    2. Concatenação:
       f_concat = Concat([f_avg, f_max]) -> [H, W, 2]

    3. Attention map:
       A = σ(Conv7x7(f_concat)) -> [H, W, 1]
       onde σ é sigmoid

    4. Output:
       F' = F ⊗ A (element-wise multiplication)

    REFERÊNCIA:
    -----------
    Inspirado em: Woo et al., 2018 - "CBAM: Convolutional Block Attention Module"
    Adaptado para imagens pequenas (28x28) com kernel 3x3 ao invés de 7x7

    JUSTIFICATIVA:
    --------------
    Para Fashion-MNIST, diferentes regiões têm importância variável:
    - Sapatos: região inferior é crítica
    - Camisetas: região superior e forma geral
    - Bolsas: forma e alças são distintivas
    """

    def __init__(self, kernel_size=3, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # Convolução para gerar o mapa de atenção
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal',
            name='attention_conv'
        )
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs):
        # Average pooling ao longo do canal
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        # Max pooling ao longo do canal
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        # Concatenar
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        # Gerar mapa de atenção
        attention_map = self.conv(concat)
        # Aplicar atenção
        return inputs * attention_map

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


class DualPathBlock(layers.Layer):
    """
    Bloco Dual-Path - Componente principal da arquitetura

    DESCRIÇÃO:
    ----------
    Processa a entrada através de dois caminhos paralelos que capturam
    diferentes escalas de informação, depois funde os resultados.

    ARQUITETURA DO BLOCO:
    ---------------------

    Input
      ├─→ Path 1 (Detail): Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU
      └─→ Path 2 (Context): Conv5x5 → BN → ReLU → DilatedConv3x3 → BN → ReLU

    Concatenate[Path1, Path2]
      → Conv1x1 (Fusion)
      → BN
      → ReLU
      → SpatialDropout
      → SpatialAttention

    CÁLCULOS DE DIMENSIONALIDADE:
    ------------------------------
    Input: [H, W, C_in]

    Path 1:
    - Conv3x3(filters): [H, W, filters]
    - Parâmetros: (3*3*C_in + 1) * filters

    Path 2:
    - Conv5x5(filters): [H, W, filters]
    - Parâmetros: (5*5*C_in + 1) * filters
    - DilatedConv3x3(filters, dilation=2): [H, W, filters]
    - Effective receptive field: 5x5
    - Parâmetros: (3*3*filters + 1) * filters

    Fusion:
    - Concat: [H, W, 2*filters]
    - Conv1x1(filters): [H, W, filters]
    - Parâmetros: (2*filters + 1) * filters

    TOTAL PARÂMETROS (aproximado):
    (9*C_in + 25*C_in + 9*filters + 2*filters) * filters + biases

    JUSTIFICATIVA:
    --------------
    - Path 1 (3x3): Captura detalhes locais e texturas finas
    - Path 2 (5x5 + dilated): Captura contexto e formas maiores
    - Fusão 1x1: Aprende combinação ótima das features
    - Atenção: Foca nas regiões mais informativas

    REFERÊNCIAS:
    ------------
    - Dilated convolutions: Yu & Koltun, 2016
    - Multi-scale processing: Szegedy et al., 2015 (Inception concept)
    - 1x1 convolutions: Lin et al., 2014 (Network in Network)
    """

    def __init__(self, filters, dilation_rate=2, **kwargs):
        super(DualPathBlock, self).__init__(**kwargs)
        self.filters = filters
        self.dilation_rate = dilation_rate

        # Path 1: Detail path (filtros pequenos)
        self.conv1_1 = layers.Conv2D(
            filters, 3, padding='same',
            kernel_regularizer=regularizers.l2(Config.L2_REG),
            kernel_initializer='he_normal',
            name='path1_conv1'
        )
        self.bn1_1 = layers.BatchNormalization(name='path1_bn1')
        self.conv1_2 = layers.Conv2D(
            filters, 3, padding='same',
            kernel_regularizer=regularizers.l2(Config.L2_REG),
            kernel_initializer='he_normal',
            name='path1_conv2'
        )
        self.bn1_2 = layers.BatchNormalization(name='path1_bn2')

        # Path 2: Context path (filtros maiores + dilatação)
        self.conv2_1 = layers.Conv2D(
            filters, 5, padding='same',
            kernel_regularizer=regularizers.l2(Config.L2_REG),
            kernel_initializer='he_normal',
            name='path2_conv1'
        )
        self.bn2_1 = layers.BatchNormalization(name='path2_bn1')
        self.conv2_2 = layers.Conv2D(
            filters, 3, padding='same',
            dilation_rate=dilation_rate,
            kernel_regularizer=regularizers.l2(Config.L2_REG),
            kernel_initializer='he_normal',
            name='path2_conv2_dilated'
        )
        self.bn2_2 = layers.BatchNormalization(name='path2_bn2')

        # Fusion layer
        self.fusion_conv = layers.Conv2D(
            filters, 1, padding='same',
            kernel_regularizer=regularizers.l2(Config.L2_REG),
            kernel_initializer='he_normal',
            name='fusion_conv'
        )
        self.fusion_bn = layers.BatchNormalization(name='fusion_bn')

        # Regularização
        self.spatial_dropout = layers.SpatialDropout2D(
            Config.SPATIAL_DROPOUT,
            name='spatial_dropout'
        )

        # Atenção espacial
        self.attention = SpatialAttention(kernel_size=3)

        self.relu = layers.ReLU()

    def call(self, inputs, training=None):
        # Path 1: Detail processing
        x1 = self.conv1_1(inputs)
        x1 = self.bn1_1(x1, training=training)
        x1 = self.relu(x1)
        x1 = self.conv1_2(x1)
        x1 = self.bn1_2(x1, training=training)
        x1 = self.relu(x1)

        # Path 2: Context processing
        x2 = self.conv2_1(inputs)
        x2 = self.bn2_1(x2, training=training)
        x2 = self.relu(x2)
        x2 = self.conv2_2(x2)
        x2 = self.bn2_2(x2, training=training)
        x2 = self.relu(x2)

        # Concatenate both paths
        x = tf.concat([x1, x2], axis=-1)

        # Fusion
        x = self.fusion_conv(x)
        x = self.fusion_bn(x, training=training)
        x = self.relu(x)

        # Regularization
        x = self.spatial_dropout(x, training=training)

        # Spatial attention
        x = self.attention(x)

        return x

    def get_config(self):
        config = super(DualPathBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'dilation_rate': self.dilation_rate
        })
        return config


# ============================================================================
# ARQUITETURA PRINCIPAL: DualPath-AttentionNet
# ============================================================================

def create_dualpath_attention_net():
    """
    Cria o modelo DualPath-AttentionNet completo

    ARQUITETURA COMPLETA:
    ---------------------

    Input (28, 28, 1)
      ↓
    [Initial Feature Extraction]
    Conv2D(32, 3x3) → BN → ReLU → SpatialDropout
      ↓ (28, 28, 32)

    [DualPath Block 1]
    DualPathBlock(64) → MaxPooling2D
      ↓ (14, 14, 64)

    [DualPath Block 2]
    DualPathBlock(128) → MaxPooling2D
      ↓ (7, 7, 128)

    [DualPath Block 3]
    DualPathBlock(256)
      ↓ (7, 7, 256)

    [Global Feature Aggregation]
    GlobalAveragePooling2D + GlobalMaxPooling2D → Concatenate
      ↓ (512,)

    [Classification Head]
    Dense(256) → BN → ReLU → Dropout(0.3)
      ↓
    Dense(128) → BN → ReLU → Dropout(0.3)
      ↓
    Dense(10, softmax)

    CÁLCULOS DE PARÂMETROS:
    -----------------------

    1. Initial Conv: (3*3*1 + 1) * 32 = 320

    2. DualPath Block 1 (filters=64, C_in=32):
       Path1: (3*3*32 + 1)*64 + (3*3*64 + 1)*64 = 55,552
       Path2: (5*5*32 + 1)*64 + (3*3*64 + 1)*64 = 88,576
       Fusion: (2*64 + 1)*64 = 8,256
       Total: ~152,384

    3. DualPath Block 2 (filters=128, C_in=64):
       Similar calculation: ~608,384

    4. DualPath Block 3 (filters=256, C_in=128):
       Similar calculation: ~2,425,856

    5. Classification Head:
       Dense1: 512*256 = 131,072
       Dense2: 256*128 = 32,768
       Dense3: 128*10 = 1,280
       Total: 165,120

    TOTAL APROXIMADO: ~3.3M parâmetros

    JUSTIFICATIVAS DAS ESCOLHAS:
    ----------------------------

    1. Progressão de filtros (32→64→128→256):
       - Aumenta gradualmente a capacidade de representação
       - Permite captura de features cada vez mais abstratas
       - Baseado em: Simonyan & Zisserman, 2015 (VGG principle)

    2. MaxPooling após blocos 1 e 2:
       - Reduz dimensionalidade espacial (28→14→7)
       - Aumenta receptive field
       - Introduz invariância a pequenas translações
       - Não usado após bloco 3 para preservar features

    3. Global Pooling (Average + Max):
       - GAP: Preserva informação média das features
       - GMP: Preserva features mais salientes
       - Concatenação: Combina ambas informações
       - Referência: Lin et al., 2014 (Network in Network)

    4. Classification Head (256→128→10):
       - Duas camadas densas para aprendizado não-linear
       - Redução gradual de dimensionalidade
       - Dropout para regularização
       - BN para estabilidade no treinamento

    5. Batch Normalization:
       - Estabiliza treinamento
       - Permite learning rates maiores
       - Reduz internal covariate shift
       - Referência: Ioffe & Szegedy, 2015

    6. Regularização (L2 + Dropout + Spatial Dropout):
       - L2: Penaliza pesos grandes, evita overfitting
       - Dropout: Força redundância nas representações
       - Spatial Dropout: Preserva coerência espacial
       - Referência: Srivastava et al., 2014 (Dropout)

    VANTAGENS PARA FASHION-MNIST:
    -----------------------------
    - Dual-path captura tanto detalhes quanto contexto
    - Atenção foca em regiões discriminativas
    - Multi-escala ajuda com variações intra-classe
    - Regularização previne overfitting (60k samples)
    - Arquitetura adequada para imagens pequenas (28x28)
    """

    inputs = keras.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, Config.IMG_CHANNELS))

    # ========================================================================
    # BLOCO INICIAL: Feature Extraction
    # ========================================================================
    # Objetivo: Extrair features básicas da imagem raw
    # Output shape: (28, 28, 32)
    x = layers.Conv2D(
        32, 3, padding='same',
        kernel_regularizer=regularizers.l2(Config.L2_REG),
        kernel_initializer='he_normal',
        name='initial_conv'
    )(inputs)
    x = layers.BatchNormalization(name='initial_bn')(x)
    x = layers.ReLU(name='initial_relu')(x)
    x = layers.SpatialDropout2D(Config.SPATIAL_DROPOUT, name='initial_dropout')(x)

    # ========================================================================
    # BLOCO 1: DualPath Block + Pooling
    # ========================================================================
    # Objetivo: Capturar features de baixo nível (edges, texturas)
    # Filters: 64
    # Output shape: (14, 14, 64)
    x = DualPathBlock(64, name='dualpath_block_1')(x)
    x = layers.MaxPooling2D(2, name='pool_1')(x)

    # ========================================================================
    # BLOCO 2: DualPath Block + Pooling
    # ========================================================================
    # Objetivo: Capturar features de médio nível (partes, padrões)
    # Filters: 128
    # Output shape: (7, 7, 128)
    x = DualPathBlock(128, name='dualpath_block_2')(x)
    x = layers.MaxPooling2D(2, name='pool_2')(x)

    # ========================================================================
    # BLOCO 3: DualPath Block (sem pooling)
    # ========================================================================
    # Objetivo: Capturar features de alto nível (formas, objetos)
    # Filters: 256
    # Output shape: (7, 7, 256)
    # Sem pooling para preservar informação espacial antes da classificação
    x = DualPathBlock(256, dilation_rate=3, name='dualpath_block_3')(x)

    # ========================================================================
    # GLOBAL FEATURE AGGREGATION
    # ========================================================================
    # Combina Global Average Pooling e Global Max Pooling
    # GAP: média das features (informação geral)
    # GMP: máximo das features (features mais salientes)
    gap = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    gmp = layers.GlobalMaxPooling2D(name='global_max_pool')(x)
    x = layers.Concatenate(name='global_concat')([gap, gmp])
    # Output shape: (512,) = 256*2

    # ========================================================================
    # CLASSIFICATION HEAD
    # ========================================================================
    # Dense Layer 1: 512 → 256
    x = layers.Dense(
        256,
        kernel_regularizer=regularizers.l2(Config.L2_REG),
        kernel_initializer='he_normal',
        name='dense_1'
    )(x)
    x = layers.BatchNormalization(name='dense_bn_1')(x)
    x = layers.ReLU(name='dense_relu_1')(x)
    x = layers.Dropout(Config.DROPOUT_RATE, name='dropout_1')(x)

    # Dense Layer 2: 256 → 128
    x = layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(Config.L2_REG),
        kernel_initializer='he_normal',
        name='dense_2'
    )(x)
    x = layers.BatchNormalization(name='dense_bn_2')(x)
    x = layers.ReLU(name='dense_relu_2')(x)
    x = layers.Dropout(Config.DROPOUT_RATE, name='dropout_2')(x)

    # Output Layer: 128 → 10 classes
    outputs = layers.Dense(
        Config.NUM_CLASSES,
        activation='softmax',
        kernel_initializer='glorot_uniform',
        name='output'
    )(x)

    # Criar modelo
    model = keras.Model(inputs=inputs, outputs=outputs, name='DualPath_AttentionNet')

    return model


# ============================================================================
# PREPARAÇÃO DOS DADOS
# ============================================================================

def load_and_preprocess_data():
    """
    Carrega e preprocessa o dataset Fashion-MNIST

    PASSOS DE PREPROCESSAMENTO:
    ---------------------------
    1. Carregamento dos dados
    2. Normalização: [0, 255] → [0, 1]
    3. Reshape: adiciona dimensão de canal
    4. One-hot encoding dos labels
    5. Split: 80% treino, 20% validação

    JUSTIFICATIVAS:
    ---------------
    - Normalização [0,1]: Estabiliza treinamento, facilita convergência
    - Validação 20%: Balanceamento entre dados de treino e validação
    - One-hot: Necessário para categorical crossentropy
    - Reshape: TensorFlow espera (H, W, C)

    Returns:
    --------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    print("Carregando Fashion-MNIST dataset...")
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    print(f"Dataset original:")
    print(f"  Treino: {X_train_full.shape}")
    print(f"  Teste: {X_test.shape}")

    # Normalização para [0, 1]
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Reshape para adicionar dimensão de canal
    X_train_full = X_train_full.reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE, 1)
    X_test = X_test.reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE, 1)

    # Split treino/validação (80/20)
    num_val = int(0.2 * len(X_train_full))
    X_val = X_train_full[-num_val:]
    y_val = y_train_full[-num_val:]
    X_train = X_train_full[:-num_val]
    y_train = y_train_full[:-num_val]

    # One-hot encoding
    y_train = to_categorical(y_train, Config.NUM_CLASSES)
    y_val = to_categorical(y_val, Config.NUM_CLASSES)
    y_test_categorical = to_categorical(y_test, Config.NUM_CLASSES)

    print(f"\nDataset processado:")
    print(f"  Treino: {X_train.shape}")
    print(f"  Validação: {X_val.shape}")
    print(f"  Teste: {X_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test, y_test_categorical)


def create_data_augmentation():
    """
    Cria pipeline de data augmentation

    TRANSFORMAÇÕES APLICADAS:
    -------------------------
    1. Rotação: ±10 graus
    2. Translação horizontal: ±10%
    3. Translação vertical: ±10%
    4. Zoom: ±10%

    JUSTIFICATIVAS:
    ---------------
    - Rotação moderada (10°): Roupas podem estar em ângulos ligeiramente diferentes
    - Translação: Objetos podem não estar perfeitamente centralizados
    - Zoom: Simula diferentes distâncias/tamanhos de objetos
    - Sem flip horizontal: Sapatos e algumas roupas têm orientação importante
    - Transformações sutis: Imagens já são pequenas (28x28)

    BENEFÍCIOS:
    -----------
    - Aumenta efetivamente o tamanho do dataset
    - Reduz overfitting
    - Melhora generalização
    - Simula variações naturais dos dados

    REFERÊNCIA:
    -----------
    - Perez & Wang, 2017 - "The Effectiveness of Data Augmentation in Image Classification"
    """
    return keras.preprocessing.image.ImageDataGenerator(
        rotation_range=Config.ROTATION_RANGE,
        width_shift_range=Config.WIDTH_SHIFT,
        height_shift_range=Config.HEIGHT_SHIFT,
        zoom_range=Config.ZOOM_RANGE,
        fill_mode='nearest'
    )


# ============================================================================
# TREINAMENTO
# ============================================================================

def train_model(model, train_data, val_data, use_augmentation=True):
    """
    Treina o modelo com callbacks e data augmentation

    CONFIGURAÇÃO DE TREINAMENTO:
    ----------------------------

    1. Optimizer: Adam
       - Learning rate: 0.001
       - Beta_1: 0.9, Beta_2: 0.999
       - Epsilon: 1e-7
       - Justificativa: Adam é robusto e adapta learning rate automaticamente
       - Referência: Kingma & Ba, 2015 - "Adam: A Method for Stochastic Optimization"

    2. Loss Function: Categorical Crossentropy
       - Justificativa: Padrão para classificação multi-classe
       - Formula: L = -Σ y_true * log(y_pred)

    3. Metrics: Accuracy
       - Justificativa: Métrica principal do desafio (4 casas decimais)

    CALLBACKS:
    ----------

    1. EarlyStopping:
       - Monitor: val_loss
       - Patience: 15 epochs
       - Restore best weights: True
       - Justificativa: Previne overfitting, economiza tempo

    2. ReduceLROnPlateau:
       - Monitor: val_loss
       - Factor: 0.5
       - Patience: 7 epochs
       - Min LR: 1e-7
       - Justificativa: Ajusta LR quando estacionar, melhora convergência

    3. ModelCheckpoint:
       - Salva melhor modelo baseado em val_accuracy
       - Justificativa: Garante que não perdemos o melhor resultado

    Parameters:
    -----------
    model: Modelo Keras
    train_data: (X_train, y_train)
    val_data: (X_val, y_val)
    use_augmentation: bool, usar data augmentation

    Returns:
    --------
    history: Histórico de treinamento
    """
    X_train, y_train = train_data
    X_val, y_val = val_data

    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Treinar com ou sem data augmentation
    if use_augmentation:
        print("\nTreinando com Data Augmentation...")
        datagen = create_data_augmentation()
        datagen.fit(X_train)

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=Config.BATCH_SIZE),
            epochs=Config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        print("\nTreinando sem Data Augmentation...")
        history = model.fit(
            X_train, y_train,
            batch_size=Config.BATCH_SIZE,
            epochs=Config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

    return history


# ============================================================================
# AVALIAÇÃO
# ============================================================================

def evaluate_model(model, test_data):
    """
    Avalia o modelo no conjunto de teste

    MÉTRICAS CALCULADAS:
    --------------------
    1. Accuracy: % de predições corretas (métrica principal do desafio)
    2. Loss: Categorical crossentropy
    3. Per-class accuracy: Accuracy para cada classe
    4. Confusion Matrix: Matriz de confusão
    5. Classification Report: Precision, Recall, F1-score por classe

    JUSTIFICATIVA:
    --------------
    - Accuracy: Métrica de avaliação do desafio (4 casas decimais)
    - Confusion matrix: Identifica quais classes são confundidas
    - Per-class metrics: Identifica classes problemáticas
    - Útil para entender pontos fortes e fracos do modelo

    Parameters:
    -----------
    model: Modelo treinado
    test_data: (X_test, y_test, y_test_categorical)

    Returns:
    --------
    results: Dict com métricas
    predictions: Predições do modelo
    """
    X_test, y_test, y_test_categorical = test_data

    print("\n" + "="*70)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("="*70)

    # Avaliar
    test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)

    print(f"\nTest Loss: {test_loss:.6f}")
    print(f"Test Accuracy: {test_accuracy:.6f} ({test_accuracy*100:.4f}%)")

    # Predições
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification Report
    report = classification_report(
        y_test, y_pred,
        target_names=Config.CLASS_NAMES,
        digits=4
    )

    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(report)

    # Accuracy por classe
    print("\n" + "="*70)
    print("ACCURACY POR CLASSE")
    print("="*70)
    for i, class_name in enumerate(Config.CLASS_NAMES):
        class_correct = cm[i, i]
        class_total = cm[i].sum()
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"{class_name:15s}: {class_acc:.4f} ({class_correct}/{class_total})")

    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return results, y_pred, y_pred_prob


def plot_training_history(history):
    """
    Plota gráficos do histórico de treinamento

    Gera 2 gráficos:
    1. Accuracy (treino vs validação)
    2. Loss (treino vs validação)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\nGráfico de treinamento salvo: training_history.png")
    plt.show()


def plot_confusion_matrix(cm, class_names):
    """
    Plota matriz de confusão
    """
    plt.figure(figsize=(12, 10))

    # Normalizar por linha (por classe verdadeira)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proporção'}
    )

    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Matriz de confusão salva: confusion_matrix.png")
    plt.show()


def visualize_predictions(model, X_test, y_test, num_samples=20):
    """
    Visualiza predições do modelo em amostras aleatórias
    """
    indices = np.random.choice(len(X_test), num_samples, replace=False)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.ravel()

    for i, idx in enumerate(indices):
        img = X_test[idx]
        true_label = y_test[idx]

        pred_prob = model.predict(img.reshape(1, 28, 28, 1), verbose=0)[0]
        pred_label = np.argmax(pred_prob)
        confidence = pred_prob[pred_label]

        axes[i].imshow(img.reshape(28, 28), cmap='gray')
        axes[i].axis('off')

        color = 'green' if pred_label == true_label else 'red'
        title = f'True: {Config.CLASS_NAMES[true_label]}\n'
        title += f'Pred: {Config.CLASS_NAMES[pred_label]}\n'
        title += f'Conf: {confidence:.2f}'
        axes[i].set_title(title, fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig('predictions_sample.png', dpi=300, bbox_inches='tight')
    print("Amostra de predições salva: predictions_sample.png")
    plt.show()


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Função principal que executa todo o pipeline
    """
    print("="*70)
    print(" DUALPATH-ATTENTIONNET - FASHION-MNIST CLASSIFIER")
    print(" DESAFIO 03e - AC3")
    print("="*70)

    # Configurar seeds para reprodutibilidade
    np.random.seed(42)
    tf.random.set_seed(42)

    # 1. Carregar dados
    print("\n[1/6] Carregando e preprocessando dados...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test, y_test_cat) = load_and_preprocess_data()

    # 2. Criar modelo
    print("\n[2/6] Criando modelo DualPath-AttentionNet...")
    model = create_dualpath_attention_net()

    # 3. Resumo do modelo
    print("\n[3/6] Arquitetura do modelo:")
    print("="*70)
    model.summary()

    # Salvar arquitetura
    with open('model_architecture.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print("\nArquitetura salva em: model_architecture.txt")

    # 4. Treinar
    print("\n[4/6] Iniciando treinamento...")
    history = train_model(
        model,
        (X_train, y_train),
        (X_val, y_val),
        use_augmentation=True
    )

    # 5. Plotar histórico
    print("\n[5/6] Gerando gráficos de treinamento...")
    plot_training_history(history)

    # 6. Avaliar
    print("\n[6/6] Avaliando modelo...")
    results, y_pred, y_pred_prob = evaluate_model(
        model,
        (X_test, y_test, y_test_cat)
    )

    # Visualizações
    print("\nGerando visualizações...")
    plot_confusion_matrix(results['confusion_matrix'], Config.CLASS_NAMES)
    visualize_predictions(model, X_test, y_test, num_samples=20)

    # Salvar resultados
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResultados salvos em: results.json")

    # Salvar modelo
    model.save('dualpath_attentionnet_final.h5')
    print("Modelo final salvo: dualpath_attentionnet_final.h5")

    print("\n" + "="*70)
    print("TREINAMENTO CONCLUÍDO!")
    print("="*70)
    print(f"\nACURÁCIA FINAL NO TESTE: {results['test_accuracy']:.6f}")
    print(f"(Com 4 casas decimais: {results['test_accuracy']*100:.4f}%)")
    print("="*70)


if __name__ == "__main__":
    main()
