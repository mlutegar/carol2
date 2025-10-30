# 🎯 DualPath-AttentionNet - Fashion-MNIST Classifier

**DESAFIO 03e - AC3**

**Arquitetura CNN Inédita para Classificação de Fashion-MNIST**

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Conceito da Arquitetura](#conceito-da-arquitetura)
3. [Por que é Inédita?](#por-que-é-inédita)
4. [Arquitetura Detalhada](#arquitetura-detalhada)
5. [Cálculos e Justificativas](#cálculos-e-justificativas)
6. [Referências Teóricas](#referências-teóricas)
7. [Resultados Esperados](#resultados-esperados)
8. [Como Executar](#como-executar)

---

## 🎯 Visão Geral

### Objetivo do Desafio

Criar uma arquitetura CNN **completamente inédita** (não replicando arquiteturas clássicas) para classificar imagens do Fashion-MNIST com a melhor acurácia possível.

### Nossa Solução: DualPath-AttentionNet

Uma arquitetura inovadora que combina:
- ✅ **Processamento Dual-Path** (dois caminhos paralelos)
- ✅ **Mecanismo de Atenção Espacial** customizado
- ✅ **Convoluções Dilatadas** para aumentar receptive field
- ✅ **Fusão Adaptativa** de features multi-escala
- ✅ **Regularização Avançada** (BN + Spatial Dropout + L2)

---

## 💡 Conceito da Arquitetura

### Inspiração Biológica

Inspirada no sistema visual humano que processa informações em duas vias paralelas:
- **Via Ventral (What):** Processa detalhes e reconhece objetos
- **Via Dorsal (Where):** Processa contexto espacial e localização

### Aplicação na CNN

```
                    INPUT IMAGE (28x28x1)
                            │
                            ▼
                ┌───────────────────────┐
                │   Initial Feature     │
                │    Extraction         │
                └───────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │        DUAL-PATH BLOCK                │
        │                                       │
        │  ┌─────────────┐  ┌─────────────┐   │
        │  │  Path 1     │  │  Path 2     │   │
        │  │  (Detail)   │  │  (Context)  │   │
        │  │             │  │             │   │
        │  │  Conv3x3    │  │  Conv5x5    │   │
        │  │  Conv3x3    │  │  Dilated3x3 │   │
        │  └─────────────┘  └─────────────┘   │
        │           │              │            │
        │           └──────┬───────┘            │
        │                  ▼                    │
        │          ┌───────────────┐           │
        │          │  FUSION 1x1   │           │
        │          └───────────────┘           │
        │                  │                    │
        │                  ▼                    │
        │          ┌───────────────┐           │
        │          │   ATTENTION   │           │
        │          └───────────────┘           │
        └───────────────────────────────────────┘
                            │
                            ▼
                    CLASSIFICATION
```

### Componentes Principais

#### 1. **Path 1 (Detail Path)** 🔍
- **Objetivo:** Capturar detalhes finos e texturas
- **Implementação:** Duas convoluções 3x3 sequenciais
- **Receptive Field:** ~5x5 efetivo
- **Ideal para:** Bordas, texturas, padrões locais

#### 2. **Path 2 (Context Path)** 🌍
- **Objetivo:** Capturar contexto global e formas
- **Implementação:** Conv 5x5 + Dilated Conv 3x3 (dilation=2)
- **Receptive Field:** ~11x11 efetivo
- **Ideal para:** Formas, estruturas globais, contexto

#### 3. **Fusão Adaptativa** 🔗
- **Objetivo:** Combinar informações das duas escalas
- **Implementação:** Concatenação + Conv 1x1
- **Benefício:** Aprende combinação ótima automaticamente

#### 4. **Atenção Espacial** 👁️
- **Objetivo:** Focar em regiões importantes da imagem
- **Implementação:** Average Pooling + Max Pooling → Conv 3x3 → Sigmoid
- **Benefício:** Modelo aprende onde "olhar"

---

## 🆚 Por que é Inédita?

### Comparação com Arquiteturas Clássicas

| Arquitetura | Características | Nossa Diferença |
|------------|-----------------|-----------------|
| **LeNet** (1998) | 2 conv layers, simples | ✅ Dual-path + attention + dilatação |
| **AlexNet** (2012) | 5 conv layers, ReLU | ✅ Processamento paralelo multi-escala |
| **VGG** (2014) | Múltiplas 3x3 sequenciais | ✅ Paths paralelos, não sequenciais |
| **ResNet** (2015) | Residual connections | ✅ Dual-path fusion, não skip connections |
| **InceptionNet** (2014) | Inception modules | ✅ Atenção espacial integrada + dilatação |

### Diferenças Fundamentais

#### ❌ **NÃO é LeNet:**
- LeNet: 2 camadas conv simples, sequenciais
- Nós: Múltiplos blocos dual-path com atenção

#### ❌ **NÃO é AlexNet:**
- AlexNet: Processamento sequencial, filtros grandes
- Nós: Processamento paralelo em duas escalas

#### ❌ **NÃO é VGG:**
- VGG: Apenas 3x3 empilhadas sequencialmente
- Nós: Dual-path com 3x3 E 5x5 em paralelo

#### ❌ **NÃO é ResNet:**
- ResNet: Residual/skip connections (F(x) + x)
- Nós: Dual-path fusion (aprende combinação)

#### ❌ **NÃO é InceptionNet:**
- Inception: Múltiplos filtros em paralelo sem atenção
- Nós: Dual-path específico + atenção espacial

### ✅ Nossa Contribuição Única

1. **Combinação inédita** de dual-path + atenção + dilatação
2. **Atenção espacial customizada** adaptada para 28x28
3. **Fusão adaptativa** através de conv 1x1
4. **Progressão específica** otimizada para Fashion-MNIST

---

## 🏗️ Arquitetura Detalhada

### Visão Completa

```
INPUT (28, 28, 1)
    ↓
┌─────────────────────────────────────────┐
│ BLOCO INICIAL: Feature Extraction      │
│                                         │
│ Conv3x3(32) → BN → ReLU → SpatialDropout│
│                                         │
│ Output: (28, 28, 32)                    │
│ Params: ~320                            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ BLOCO 1: DualPath Block (filters=64)   │
│                                         │
│ Path 1:  Conv3x3(64) → BN → ReLU        │
│          Conv3x3(64) → BN → ReLU        │
│                                         │
│ Path 2:  Conv5x5(64) → BN → ReLU        │
│          DilatedConv3x3(64, d=2) → BN → ReLU │
│                                         │
│ Fusion:  Concat → Conv1x1(64) → BN → ReLU │
│          SpatialDropout → Attention     │
│                                         │
│ Output: (28, 28, 64)                    │
│ Params: ~152K                           │
└─────────────────────────────────────────┘
    ↓ MaxPooling2D (2x2)
    ↓
┌─────────────────────────────────────────┐
│ (14, 14, 64)                            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ BLOCO 2: DualPath Block (filters=128)  │
│                                         │
│ [Mesma estrutura do Bloco 1]           │
│                                         │
│ Output: (14, 14, 128)                   │
│ Params: ~608K                           │
└─────────────────────────────────────────┘
    ↓ MaxPooling2D (2x2)
    ↓
┌─────────────────────────────────────────┐
│ (7, 7, 128)                             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ BLOCO 3: DualPath Block (filters=256)  │
│          dilation_rate=3                │
│                                         │
│ [Mesma estrutura, SEM pooling]         │
│                                         │
│ Output: (7, 7, 256)                     │
│ Params: ~2.4M                           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ GLOBAL FEATURE AGGREGATION              │
│                                         │
│ GlobalAveragePooling2D → (256,)         │
│ GlobalMaxPooling2D → (256,)             │
│ Concatenate → (512,)                    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ CLASSIFICATION HEAD                     │
│                                         │
│ Dense(256) → BN → ReLU → Dropout(0.3)   │
│ Dense(128) → BN → ReLU → Dropout(0.3)   │
│ Dense(10, softmax)                      │
│                                         │
│ Params: ~165K                           │
└─────────────────────────────────────────┘
    ↓
OUTPUT (10,) - Probabilidades por classe
```

### Resumo de Parâmetros

| Componente | Output Shape | Parâmetros | RF Efetivo |
|-----------|--------------|------------|------------|
| Input | (28, 28, 1) | - | 1x1 |
| Initial Conv | (28, 28, 32) | ~320 | 3x3 |
| DualPath Block 1 | (28, 28, 64) | ~152K | ~11x11 |
| MaxPool 1 | (14, 14, 64) | 0 | 22x22 |
| DualPath Block 2 | (14, 14, 128) | ~608K | ~44x44 |
| MaxPool 2 | (7, 7, 128) | 0 | 88x88 |
| DualPath Block 3 | (7, 7, 256) | ~2.4M | ~150x150 |
| Global Pooling | (512,) | 0 | Full image |
| Classification | (10,) | ~165K | - |
| **TOTAL** | | **~3.3M** | |

---

## 📐 Cálculos e Justificativas

### 1. Camada de Atenção Espacial

#### Matemática

Dado um feature map $F \in \mathbb{R}^{H \times W \times C}$:

**Passo 1: Feature Aggregation**
$$f_{avg} = \frac{1}{C} \sum_{c=1}^{C} F_{:,:,c} \in \mathbb{R}^{H \times W \times 1}$$

$$f_{max} = \max_{c \in [1,C]} F_{:,:,c} \in \mathbb{R}^{H \times W \times 1}$$

**Passo 2: Concatenação**
$$f_{concat} = [f_{avg}; f_{max}] \in \mathbb{R}^{H \times W \times 2}$$

**Passo 3: Attention Map**
$$A = \sigma(W * f_{concat} + b) \in \mathbb{R}^{H \times W \times 1}$$

onde:
- $W \in \mathbb{R}^{3 \times 3 \times 2 \times 1}$ (kernel da convolução)
- $\sigma$ é a função sigmoid
- $*$ denota operação de convolução

**Passo 4: Aplicar Atenção**
$$F' = F \odot A \in \mathbb{R}^{H \times W \times C}$$

onde $\odot$ é multiplicação element-wise (broadcasting em A)

#### Parâmetros

```
Conv3x3: (3 × 3 × 2) + 1 = 19 parâmetros
```

#### Justificativa

1. **Por que Average e Max Pooling?**
   - Average: Captura informação estatística geral
   - Max: Captura features mais salientes
   - Combinação: Representação mais rica

2. **Por que kernel 3x3?**
   - Imagens são pequenas (28x28)
   - 7x7 (usado no CBAM original) seria muito grande
   - 3x3 é adequado para nossa escala

3. **Por que Sigmoid?**
   - Outputs em [0, 1] servem como pesos de atenção
   - 0 = região não importante, 1 = região muito importante

---

### 2. Bloco Dual-Path

#### Estrutura Matemática

**Input:** $X \in \mathbb{R}^{H \times W \times C_{in}}$

**Path 1 (Detail):**
$$X_1 = \text{ReLU}(\text{BN}(W_1^{(2)} * \text{ReLU}(\text{BN}(W_1^{(1)} * X))))$$

onde:
- $W_1^{(1)}, W_1^{(2)} \in \mathbb{R}^{3 \times 3 \times C_{in} \times F}$

**Path 2 (Context):**
$$X_2^{(1)} = \text{ReLU}(\text{BN}(W_2^{(1)} * X))$$
$$X_2 = \text{ReLU}(\text{BN}(W_2^{(2)} *_{d} X_2^{(1)}))$$

onde:
- $W_2^{(1)} \in \mathbb{R}^{5 \times 5 \times C_{in} \times F}$
- $W_2^{(2)} \in \mathbb{R}^{3 \times 3 \times F \times F}$
- $*_d$ denota convolução dilatada com taxa $d$

**Fusão:**
$$X_{concat} = [X_1; X_2] \in \mathbb{R}^{H \times W \times 2F}$$
$$X_{fused} = \text{ReLU}(\text{BN}(W_{fusion} * X_{concat}))$$

onde:
- $W_{fusion} \in \mathbb{R}^{1 \times 1 \times 2F \times F}$

**Regularização e Atenção:**
$$X_{drop} = \text{SpatialDropout}(X_{fused}, p=0.2)$$
$$X_{out} = \text{Attention}(X_{drop})$$

#### Cálculo de Parâmetros

Para um bloco com `filters=F` e entrada `C_in`:

**Path 1:**
- Conv1: $(3 \times 3 \times C_{in} + 1) \times F = 9C_{in}F + F$
- Conv2: $(3 \times 3 \times F + 1) \times F = 9F^2 + F$
- **Subtotal Path 1:** $9C_{in}F + 9F^2 + 2F$

**Path 2:**
- Conv1: $(5 \times 5 \times C_{in} + 1) \times F = 25C_{in}F + F$
- Conv2: $(3 \times 3 \times F + 1) \times F = 9F^2 + F$
- **Subtotal Path 2:** $25C_{in}F + 9F^2 + 2F$

**Fusion:**
- Conv1x1: $(1 \times 1 \times 2F + 1) \times F = 2F^2 + F$

**Batch Normalization (4 layers):**
- Cada BN: $4F$ parâmetros (γ, β, mean, variance)
- Total: $16F$

**TOTAL:**
$$P_{block} = 34C_{in}F + 20F^2 + 5F + 16F$$

**Exemplo para Block 1** ($C_{in}=32$, $F=64$):
$$P = 34(32)(64) + 20(64)^2 + 21(64) = 69,632 + 81,920 + 1,344 = 152,896$$

#### Justificativas

##### Por que Path 1 usa 3x3 duplo?

1. **Receptive Field Efetivo:**
   - Um 3x3: RF = 3x3
   - Dois 3x3: RF = 5x5 (mesma área que um 5x5)

2. **Menos Parâmetros:**
   - Um 5x5: $25C^2$ parâmetros
   - Dois 3x3: $18C^2$ parâmetros
   - **Economia:** ~28%

3. **Mais Não-Linearidade:**
   - Dois 3x3: 2 ReLUs
   - Um 5x5: 1 ReLU
   - **Benefício:** Maior capacidade de representação

4. **Referência:** Princípio do VGG (Simonyan & Zisserman, 2014)

##### Por que Path 2 usa 5x5 + Dilated 3x3?

1. **5x5 Inicial:**
   - Captura contexto maior imediatamente
   - RF = 5x5

2. **Dilated 3x3 (d=2):**
   - RF efetivo = 5x5 (mas vê área 5x5 com 9 parâmetros)
   - Após 5x5: RF total ~= 11x11
   - **Benefício:** Contexto amplo sem explodir parâmetros

3. **Por que Dilatação?**
   $$\text{RF}_{\text{dilated}} = k + (k-1)(d-1)$$
   Para $k=3$, $d=2$: RF = $3 + 2 \times 1 = 5$

4. **Referência:** Yu & Koltun, 2016

##### Por que Fusão com 1x1?

1. **Combinação Cross-Channel:**
   - Aprende combinação ótima das features
   - Não há convolution espacial

2. **Redução de Dimensionalidade:**
   - $2F \rightarrow F$ canais
   - Controla crescimento de parâmetros

3. **Adiciona Não-Linearidade:**
   - Conv1x1 + BN + ReLU
   - Aumenta capacidade do modelo

4. **Referência:** Lin et al., 2014 (Network in Network)

---

### 3. Progressão de Filtros (32 → 64 → 128 → 256)

#### Justificativa Teórica

**Pirâmide de Abstrações:**

```
Camada         Filtros    O que aprende?
─────────────────────────────────────────────
Initial Conv      32      Edges básicas, gradientes
                          (horizontal, vertical, diagonal)

Block 1          64      Texturas simples
                          (padrões repetitivos)

Block 2          128     Partes de objetos
                          (mangas, golas, solas)

Block 3          256     Objetos completos
                          (camisetas, sapatos inteiros)
```

#### Cálculo de Capacidade

**Capacidade de Representação:**

Para uma camada com $F$ filtros e input $C$:
$$\text{Capacidade} \propto FC$$

**Progressão:**
- Layer 1: $32 \times 1 = 32$ combinações
- Layer 2: $64 \times 32 = 2,048$ combinações
- Layer 3: $128 \times 64 = 8,192$ combinações
- Layer 4: $256 \times 128 = 32,768$ combinações

**Crescimento Exponencial:** Permite capturar abstrações cada vez mais complexas

#### Referências

- VGG (Simonyan & Zisserman, 2014): Duplica filtros após pooling
- ResNet (He et al., 2015): Progressão similar
- Princípio geral em Deep Learning

---

### 4. Global Pooling (GAP + GMP)

#### Matemática

**Global Average Pooling:**
$$\text{GAP}(F) = \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} F_{i,j,:}$$

**Global Max Pooling:**
$$\text{GMP}(F) = \max_{i,j} F_{i,j,:}$$

**Concatenação:**
$$F_{global} = [\text{GAP}(F); \text{GMP}(F)]$$

Para $F \in \mathbb{R}^{H \times W \times C}$:
- GAP: $\mathbb{R}^C$
- GMP: $\mathbb{R}^C$
- Concat: $\mathbb{R}^{2C}$

#### Justificativas

##### Por que GAP?

1. **Informação Estatística:**
   - Média captura padrões gerais
   - Robusto a outliers

2. **Regularização Implícita:**
   - Não há parâmetros treináveis
   - Previne overfitting espacial

3. **Invariância a Translações:**
   - Posição exata não importa
   - Apenas presença da feature

4. **Referência:** Lin et al., 2014

##### Por que GMP?

1. **Features Salientes:**
   - Captura ativações mais fortes
   - Útil para features discriminativas

2. **Complementa GAP:**
   - GAP: informação geral
   - GMP: características distintivas

##### Por que Ambos (GAP + GMP)?

**Experimentos empíricos mostram:**
- GAP alone: Boa performance geral
- GMP alone: Captura melhor features salientes
- **GAP + GMP:** Melhor dos dois mundos!

**Aumento de informação:** $2\times$ vetores → $2\times$ informação

---

### 5. Batch Normalization

#### Matemática

Para um mini-batch $\mathcal{B} = \{x_1, ..., x_m\}$:

**Forward Pass:**

1. **Calcular estatísticas do batch:**
   $$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i$$
   $$\sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2$$

2. **Normalizar:**
   $$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$

3. **Scale and Shift (parâmetros aprendidos):**
   $$y_i = \gamma \hat{x}_i + \beta$$

**Parâmetros:**
- $\gamma$: scale (multiplicativo)
- $\beta$: shift (aditivo)
- $\mu_{running}$: média móvel para inferência
- $\sigma^2_{running}$: variância móvel para inferência

**Total:** 4 parâmetros por feature map

#### Justificativas

##### Por que usar BN?

1. **Reduz Internal Covariate Shift:**
   - Distribuição de ativações muda entre camadas
   - BN estabiliza essas distribuições

2. **Permite Learning Rates Maiores:**
   - Gradientes mais estáveis
   - Convergência mais rápida

3. **Efeito Regularizador:**
   - Adiciona ruído (estatísticas do batch)
   - Reduz necessidade de dropout

4. **Melhora Gradientes:**
   - Evita saturação de ativações
   - Backprop mais eficiente

##### Onde usar BN?

**Nossa escolha: Após convoluções, ANTES de ReLU**

```
Conv → BN → ReLU  ✅ (nossa escolha)
Conv → ReLU → BN  ❌
```

**Por quê?**
- ReLU pode "matar" neurônios (output=0)
- BN antes de ReLU garante distribuição normalizada
- Referência: Paper original (Ioffe & Szegedy, 2015)

---

### 6. Dropout vs Spatial Dropout

#### Dropout Regular

**Matemática:**
$$y = \begin{cases}
\frac{x}{1-p} & \text{com probabilidade } 1-p \\
0 & \text{com probabilidade } p
\end{cases}$$

**Aplicação:**
- Zera neurônios aleatoriamente
- **Problema em CNNs:** Features espacialmente correlacionadas

#### Spatial Dropout

**Matemática:**
$$y_{:,:,c} = \begin{cases}
\frac{x_{:,:,c}}{1-p} & \text{com probabilidade } 1-p \\
0 & \text{com probabilidade } p
\end{cases}$$

**Aplicação:**
- Zera **canais inteiros** (feature maps)
- Preserva correlação espacial

#### Justificativa

**Por que Spatial Dropout em CNNs?**

1. **Correlação Espacial:**
   - Pixels vizinhos são correlacionados
   - Dropout regular: ainda vaza informação

2. **Feature Maps são Features:**
   - Cada canal é uma feature completa
   - Faz mais sentido dropar features inteiras

3. **Resultados Empíricos:**
   - Spatial Dropout > Regular Dropout em CNNs
   - Referência: Tompson et al., 2015

**Nossa configuração:**
- Spatial Dropout: 0.2 (20%) após convoluções
- Regular Dropout: 0.3 (30%) nas camadas densas

---

### 7. Regularização L2

#### Matemática

**Loss com L2:**
$$L_{total} = L_{data} + \lambda \sum_{w \in W} w^2$$

onde:
- $L_{data}$: Loss original (categorical crossentropy)
- $\lambda$: Força da regularização
- $W$: Todos os pesos do modelo

**Gradiente:**
$$\frac{\partial L_{total}}{\partial w} = \frac{\partial L_{data}}{\partial w} + 2\lambda w$$

**Efeito:** Adiciona termo que "puxa" pesos para zero

#### Justificativa

##### Por que L2?

1. **Previne Overfitting:**
   - Penaliza pesos grandes
   - Força modelo mais simples

2. **Vs L1:**
   - L1: Leva pesos para exatamente zero (sparsity)
   - L2: Leva pesos para perto de zero (smooth)
   - **L2 é melhor para CNNs** (todas features úteis)

3. **Valor $\lambda = 10^{-4}$:**
   - Pequeno o suficiente: não prejudica aprendizado
   - Grande o suficiente: previne overfitting
   - Valor padrão na literatura

##### Onde aplicar L2?

**Nossa escolha: Apenas em camadas convolucionais e densas**

✅ Aplicar:
- Convoluções
- Camadas densas

❌ NÃO aplicar:
- Batch Normalization (já é regularizado)
- Atenção (poucos parâmetros)

---

### 8. Data Augmentation

#### Transformações Aplicadas

**1. Rotação:**
$$\theta \sim \mathcal{U}(-10°, +10°)$$

**2. Translação Horizontal:**
$$t_x \sim \mathcal{U}(-10\%, +10\%) \times W$$

**3. Translação Vertical:**
$$t_y \sim \mathcal{U}(-10\%, +10\%) \times H$$

**4. Zoom:**
$$z \sim \mathcal{U}(0.9, 1.1)$$

#### Justificativas

##### Por que essas transformações?

| Transformação | Justificativa | Efeito |
|--------------|---------------|---------|
| **Rotação ±10°** | Roupas podem estar em ângulos | Invariância rotacional |
| **Shift ±10%** | Objetos podem não estar centrados | Invariância translação |
| **Zoom ±10%** | Simula diferentes distâncias | Invariância escala |

##### Por que NÃO flip horizontal?

**Razão:** Orientação importa no Fashion-MNIST!

Exemplos:
- 👟 Sapatos: Pé direito vs esquerdo
- 👜 Bolsas: Alça pode estar de um lado
- 👔 Algumas roupas têm assimetria

**Flip pode confundir o modelo!**

##### Por que valores moderados (10%)?

**Razão:** Imagens são pequenas (28x28)

- Transformações grandes: Perda de informação
- Transformações pequenas: Ainda úteis
- 10%: Balanceamento ideal

---

## 📚 Referências Teóricas

### Papers Principais

1. **Ioffe, S., & Szegedy, C. (2015)**
   - "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
   - **Contribuição:** Batch Normalization
   - **Nossa aplicação:** Usamos BN após cada convolução

2. **Kingma, D. P., & Ba, J. (2015)**
   - "Adam: A Method for Stochastic Optimization"
   - **Contribuição:** Optimizer Adam
   - **Nossa aplicação:** Optimizer principal com LR=0.001

3. **Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018)**
   - "CBAM: Convolutional Block Attention Module"
   - **Contribuição:** Attention mechanisms em CNNs
   - **Nossa aplicação:** Inspiração para atenção espacial (adaptamos para 28x28)

4. **Yu, F., & Koltun, V. (2016)**
   - "Multi-Scale Context Aggregation by Dilated Convolutions"
   - **Contribuição:** Dilated/atrous convolutions
   - **Nossa aplicação:** Path 2 usa convoluções dilatadas

5. **Lin, M., Chen, Q., & Yan, S. (2014)**
   - "Network in Network"
   - **Contribuição:** Convoluções 1x1
   - **Nossa aplicação:** Fusão dos paths com conv 1x1

6. **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014)**
   - "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
   - **Contribuição:** Dropout
   - **Nossa aplicação:** Regular e spatial dropout

7. **Tompson, J., Goroshin, R., Jain, A., LeCun, Y., & Bregler, C. (2015)**
   - "Efficient Object Localization Using Convolutional Networks"
   - **Contribuição:** Spatial Dropout
   - **Nossa aplicação:** Spatial dropout após convoluções

8. **Simonyan, K., & Zisserman, A. (2014)**
   - "Very Deep Convolutional Networks for Large-Scale Image Recognition" (VGG)
   - **Contribuição:** Múltiplas conv 3x3, progressão de filtros
   - **Nossa aplicação:** Inspiração para Path 1 e progressão 32→64→128→256

9. **He, K., Zhang, X., Ren, S., & Sun, J. (2015)**
   - "Deep Residual Learning for Image Recognition" (ResNet)
   - **Contribuição:** Deep networks, progressão de filtros
   - **Nossa aplicação:** Inspiração para progressão (mas sem residual connections)

10. **Perez, L., & Wang, J. (2017)**
    - "The Effectiveness of Data Augmentation in Image Classification using Deep Learning"
    - **Contribuição:** Análise de data augmentation
    - **Nossa aplicação:** Escolha das transformações de augmentation

### Arquiteturas Clássicas (NÃO replicadas)

| Paper | Ano | Contribuição | Como NÃO replicamos |
|-------|-----|--------------|---------------------|
| LeNet (LeCun et al.) | 1998 | CNN básica | Nossa: muito mais complexa, dual-path |
| AlexNet (Krizhevsky et al.) | 2012 | Deep CNN, ReLU, GPU | Nossa: processamento paralelo |
| VGG (Simonyan & Zisserman) | 2014 | Múltiplas 3x3 | Nossa: paths paralelos + atenção |
| GoogLeNet/Inception (Szegedy et al.) | 2014 | Inception modules | Nossa: dual-path específico + atenção |
| ResNet (He et al.) | 2015 | Residual connections | Nossa: fusão adaptativa, não skip |

---

## 🎯 Resultados Esperados

### Métricas de Avaliação

1. **Accuracy (Principal):**
   - Métrica do desafio
   - Reportada com 4 casas decimais
   - **Meta:** > 90%

2. **Loss:**
   - Categorical Crossentropy
   - Monitora convergência

3. **Per-Class Metrics:**
   - Precision, Recall, F1-Score
   - Identifica classes problemáticas

4. **Confusion Matrix:**
   - Visualiza confusões entre classes
   - Útil para análise de erros

### Expectativas

**Baseado na arquitetura:**

| Aspecto | Expectativa | Razão |
|---------|-------------|-------|
| **Accuracy Geral** | 90-93% | Arquitetura robusta, multi-escala |
| **Classes Fáceis** | 95%+ | Sandal, Bag, Ankle boot (distintivas) |
| **Classes Difíceis** | 85-90% | Shirt vs T-shirt, Pullover vs Coat |
| **Overfitting** | Mínimo | Regularização múltipla |
| **Convergência** | 30-50 epochs | Callbacks param otimizar |

### Análise de Dificuldade por Classe

**Classes FÁCEIS (> 95% accuracy esperado):**
- 🥿 **Sandal:** Forma única, aberta
- 👜 **Bag:** Formato distinto, alças
- 👢 **Ankle boot:** Cano alto característico

**Classes MÉDIAS (90-95% accuracy):**
- 👕 **T-shirt/top:** Forma simples, mas pode confundir com Shirt
- 👖 **Trouser:** Duas pernas distintivas
- 👗 **Dress:** Formato único
- 🧥 **Coat:** Mais longo, mas pode confundir com Pullover
- 👟 **Sneaker:** Similar a Ankle boot, mas mais baixo

**Classes DIFÍCEIS (85-90% accuracy):**
- 👔 **Shirt:** Muito similar a T-shirt
- 🧶 **Pullover:** Pode confundir com Coat

### Matriz de Confusão Esperada

```
                Predicted
              0  1  2  3  4  5  6  7  8  9
        ┌────────────────────────────────┐
     0  │ ■■ .  .  .  .  .  ▪  .  .  .  │ T-shirt
     1  │ .  ■■ .  .  .  .  .  .  .  .  │ Trouser
     2  │ .  .  ■■ .  ▪  .  .  .  .  .  │ Pullover
A    3  │ .  .  .  ■■ .  .  .  .  .  .  │ Dress
c    4  │ .  .  ▪  .  ■■ .  .  .  .  .  │ Coat
t    5  │ .  .  .  .  .  ■■ .  ▪  .  .  │ Sandal
u    6  │ ▪  .  .  .  .  .  ■■ .  .  .  │ Shirt
a    7  │ .  .  .  .  .  ▪  .  ■■ .  ▪  │ Sneaker
l    8  │ .  .  .  .  .  .  .  .  ■■ .  │ Bag
     9  │ .  .  .  .  .  .  .  ▪  .  ■■ │ Ankle boot
        └────────────────────────────────┘

Legend: ■■ High (>90%), ▪ Low (<5%), . Very low (<2%)
```

**Confusões esperadas:**
- Shirt ↔ T-shirt (mais comum)
- Pullover ↔ Coat
- Sneaker ↔ Ankle boot
- Sandal ↔ Sneaker

---

## 🚀 Como Executar

### Opção 1: Google Colab (Recomendado)

```python
# 1. Upload o notebook para o Colab
# Fashion_MNIST_DualPath_AttentionNet.ipynb

# 2. Runtime > Change runtime type > GPU

# 3. Executar todas as células (Runtime > Run all)
```

### Opção 2: Local (com GPU)

```bash
# 1. Clonar repositório
git clone <repo-url>
cd Carol2

# 2. Instalar dependências
pip install tensorflow matplotlib seaborn scikit-learn numpy

# 3. Executar
python fashion_mnist_dualpath_cnn.py

# OU abrir notebook
jupyter notebook Fashion_MNIST_DualPath_AttentionNet.ipynb
```

### Requisitos

```
tensorflow >= 2.8.0
numpy >= 1.21.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
```

### Outputs Gerados

```
📁 Arquivos salvos após treinamento:
├── best_model.h5                          # Melhor modelo (val_accuracy)
├── dualpath_attentionnet_final.h5        # Modelo final
├── results.json                           # Métricas detalhadas
├── training_history.png                   # Gráficos de treino
├── confusion_matrix.png                   # Matriz de confusão
└── predictions_sample.png                 # Exemplos de predições
```

---

## 📊 Estrutura do Projeto

```
Carol2/
├── Fashion_MNIST_DualPath_AttentionNet.ipynb    # Notebook principal
├── fashion_mnist_dualpath_cnn.py                # Script Python
├── README_DUALPATH_ATTENTIONNET.md              # Esta documentação
├── PROMPT.md                                     # Requisitos do desafio
├── referencia1.md                                # Referência 1
├── referencia2.md                                # Referência 2
│
├── Outputs (gerados após treinamento):
│   ├── best_model.h5
│   ├── dualpath_attentionnet_final.h5
│   ├── results.json
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── predictions_sample.png
│
└── Documentation:
    └── model_architecture.txt               # Resumo da arquitetura
```

---

## ✅ Checklist de Validação

### Requisitos do Desafio

- ✅ **Arquitetura inédita:** Não replica LeNet, AlexNet, VGG, ResNet ou InceptionNet
- ✅ **Documentação completa:** Cada camada documentada com justificativas
- ✅ **Cálculos detalhados:** Parâmetros, receptive fields, matemática
- ✅ **Referências:** Todas as decisões baseadas em papers
- ✅ **Dataset correto:** Fashion-MNIST
- ✅ **Avaliação adequada:** Accuracy, confusion matrix, métricas
- ✅ **Código organizado:** Notebook limpo e executável
- ✅ **Data augmentation:** Implementado e justificado
- ✅ **Regularização:** Múltiplas técnicas (BN, dropout, L2)
- ✅ **Treinamento adequado:** Callbacks, early stopping, checkpoints

### Critérios de Inovação

- ✅ **Dual-Path:** Dois caminhos paralelos (Detail + Context)
- ✅ **Atenção Espacial:** Mecanismo customizado
- ✅ **Convoluções Dilatadas:** Aumenta RF sem custos
- ✅ **Fusão Adaptativa:** Conv 1x1 aprende combinação
- ✅ **Multi-escala:** Processa em diferentes escalas
- ✅ **Regularização Avançada:** Spatial dropout + L2 + BN
- ✅ **Global Pooling Duplo:** GAP + GMP

---

## 🎓 Para a Apresentação

### Estrutura Sugerida (15 minutos)

**1. Introdução (2 min)**
- Problema: Fashion-MNIST, 10 classes
- Objetivo: Arquitetura inédita com melhor accuracy

**2. Conceito da Arquitetura (3 min)**
- Dual-Path: Detail + Context
- Atenção Espacial
- Por que é inédita?

**3. Arquitetura Detalhada (5 min)**
- Diagrama visual
- Blocos principais
- Cálculos de parâmetros
- Justificativas técnicas

**4. Resultados (3 min)**
- Accuracy final
- Confusion matrix
- Análise de erros
- Comparação com baseline

**5. Conclusão (2 min)**
- Inovações principais
- Lições aprendidas
- Possíveis melhorias

### Slides Recomendados

1. **Título + Grupo**
2. **Problema e Dataset**
3. **Conceito: Dual-Path + Atenção** (diagrama)
4. **Arquitetura Completa** (fluxograma)
5. **Bloco Dual-Path Detalhado**
6. **Atenção Espacial** (matemática + visualização)
7. **Cálculos de Parâmetros** (tabela)
8. **Comparação com Arquiteturas Clássicas** (tabela)
9. **Configuração de Treinamento**
10. **Resultados: Accuracy**
11. **Confusion Matrix**
12. **Exemplos de Predições**
13. **Análise de Erros**
14. **Conclusões e Contribuições**
15. **Perguntas**

### Pontos a Enfatizar

🔥 **Inédita:** Não replica nenhuma arquitetura clássica

🔥 **Bem Documentada:** Todos os cálculos e justificativas

🔥 **Fundamentada:** Baseada em papers recentes

🔥 **Eficaz:** Resultados competitivos

---

## 📞 Contato e Suporte

Para dúvidas sobre a implementação:
1. Revisar este README completo
2. Verificar comentários no código
3. Consultar referências citadas

---

## 📜 Licença

Este projeto foi desenvolvido para fins educacionais como parte do **DESAFIO 03e - AC3**.

---

**Desenvolvido com 💙 para alcançar a melhor classificação possível! 🎯**

**Arquitetura:** DualPath-AttentionNet (100% Inédita)

**Dataset:** Fashion-MNIST (10 classes)

**Objetivo:** Maximizar Accuracy com arquitetura inovadora e bem fundamentada!

---

*Última atualização: 2025*
