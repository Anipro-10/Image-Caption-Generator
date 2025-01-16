# Image Captioning Project

This project implements an image captioning system using VGG16 and LSTM networks to generate natural language descriptions of images.

## Description

The system takes an image as input and generates descriptive captions using:
- VGG16 for image feature extraction
- LSTM network for text generation
- Embedding layer for text processing
- Dense layers for feature combination

## Dependencies

```
tensorflow
numpy
pillow
tqdm
nltk
matplotlib
```

## Dataset

The project uses the Flickr8k dataset organized as:
```
BASE_DIR/
    ├── Images/         
    └── captions.txt    
```

## Code Structure

1. **Feature Extraction**
```python
model1 = VGG16()
model1 = Model(model1.inputs, model1.layers[-2].output)
```
Extracts image features using VGG16 network.

2. **Data Processing**
```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
```
Processes and tokenizes image captions.

3. **Model Architecture**
```python
#image features
inputs1 = Input(shape=(4096,), name="image")
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

#sequence features
inputs2 = Input(shape=(max_length,), name="text")
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder 
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
```

4. **Training**
```python
epochs = 20
batch_size = 32
steps = len(train) // batch_size
```

5. **Caption Generation**
```python
def generate_caption(image_name):
    # Generate captions for new images
```

## Usage

1. Extract Features:
```python
features = {}
for img_name in os.listdir(directory):
    image = load_img(img_path, target_size=(224, 224))
    feature = model1.predict(image, verbose=0)
    features[image_id] = feature
```

2. Train Model:
```python
model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
```

3. Generate Captions:
```python
generate_caption('image.jpg')
```

## Results

The model is evaluated using:
- BLEU-1 score for individual word accuracy: 0.52
- BLEU-2 score for phrase accuracy: 0.29

Example output:
```python
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
```

## Model Flow

1. Image → VGG16 → Feature Vector (4096)
2. Caption → Tokenization → Sequence
3. Combined Features → LSTM → Generated Caption


