import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import StringLookup
from tensorflow.keras import layers

# Disable eager execution for using tf.compat.v1.keras.backend.ctc_decode
tf.compat.v1.disable_eager_execution()


### 1. SETUP AND DATA LOADING ###

# --- You will need to download the IAM dataset first ---
# Create a folder named 'data' and inside it, extract the 'words' folder and 'words.txt' from the IAM dataset zip.
# Or run this in a shell to download and prepare:
# !wget -q https://git.io/J0fjL -O IAM_Words.zip
# !unzip -qq IAM_Words.zip
# !mkdir -p data/words
# !tar -xf IAM_Words/words.tgz -C data/words
# !mv IAM_Words/words.txt data/

# Path to the dataset
base_path = "data"
words_list_path = os.path.join(base_path, "words.txt")
words_folder_path = os.path.join(base_path, "words")

# Check if the necessary files/folders exist
if not os.path.exists(words_list_path) or not os.path.exists(words_folder_path):
    print("Please download and set up the IAM Handwriting Dataset.")
    print("Ensure 'words.txt' is in 'data/' and the 'words' image folder is in 'data/'.")
    exit()

# Get a list of all image paths and their corresponding labels
image_paths = []
labels = []
with open(words_list_path, "r") as f:
    for line in f:
        # Skip comment lines
        if line.startswith("#"):
            continue
        
        line_split = line.strip().split(" ")
        # We need at least 9 components for a valid line
        if len(line_split) < 9:
            continue
        
        # The file name is the first component
        file_name = line_split[0]
        # The handwritten word is the last component
        label = line_split[-1]
        
        # Build the full image path
        # e.g., a01/a01-000u/a01-000u-00-00.png
        folder1 = file_name.split("-")[0]
        folder2 = "-".join(file_name.split("-")[0:2])
        img_path = os.path.join(words_folder_path, folder1, folder2, f"{file_name}.png")

        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(label)

print(f"Total images found: {len(image_paths)}")
print(f"Total labels found: {len(labels)}")
print(f"Example image path: {image_paths[0]}")
print(f"Example label: {labels[0]}")


# Split data into training, validation, and test sets
split_idx = int(0.9 * len(image_paths))
train_samples = image_paths[:split_idx]
train_labels = labels[:split_idx]

validation_samples = image_paths[split_idx:]
validation_labels = labels[split_idx:]

print(f"Training samples: {len(train_samples)}")
print(f"Validation samples: {len(validation_samples)}")

# Create character vocabulary
all_chars = sorted(list(set("".join(labels))))
char_to_num = StringLookup(vocabulary=list(all_chars), mask_token=None)
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

print(f"Vocabulary size: {char_to_num.vocabulary_size()}")
print(f"Vocabulary: {char_to_num.get_vocabulary()}")

# Set image dimensions
IMG_WIDTH = 200
IMG_HEIGHT = 50

# Data preprocessing and generator
def encode_single_sample(img_path, label):
    # 1. Read image and convert to grayscale
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    # 2. Resize to target dimensions
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    
    # 3. Transpose for CTC compatibility (width becomes the time dimension)
    img = tf.transpose(img, perm=[1, 0, 2])
    
    # 4. Map characters in label to numbers
    label_encoded = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    
    return {"image": img, "label": label_encoded}

class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, img_width=200, img_height=50):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.char_to_num = StringLookup(vocabulary=list(all_chars), mask_token=None)
        self.num_samples = len(image_paths)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.image_paths[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]
        
        X, y = self.__data_generation(batch_images, batch_labels)
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(self.num_samples)
        np.random.shuffle(self.indices)

    def __data_generation(self, batch_images, batch_labels):
        # Prepare batch data for the model
        images = np.ones((self.batch_size, self.img_width, self.img_height, 1), dtype=np.float32)
        labels = np.zeros((self.batch_size, 50), dtype=np.float32) # Max label length placeholder
        input_length = np.ones((self.batch_size, 1)) * (self.img_width // 4 - 2) # Downsampled width
        label_length = np.zeros((self.batch_size, 1), dtype=np.int64)

        for i, (img_path, label_text) in enumerate(zip(batch_images, batch_labels)):
            img = tf.io.read_file(img_path)
            img = tf.io.decode_png(img, channels=1)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [self.img_height, self.img_width])
            img = tf.transpose(img, perm=[1, 0, 2])
            images[i, :, :, :] = img

            encoded_label = self.char_to_num(tf.strings.unicode_split(label_text, input_encoding="UTF-8"))
            labels[i, 0:len(encoded_label)] = encoded_label
            label_length[i] = len(encoded_label)
        
        inputs = {
            'image': images,
            'label': labels,
            'input_length': input_length,
            'label_length': label_length,
        }
        return inputs, labels


### 2. BUILD THE CRNN MODEL ###

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        # Compute the training-time loss value
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

def build_model():
    # Inputs to the model
    input_img = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name="image")
    labels = layers.Input(name="label", shape=(None,))
    input_length = layers.Input(name="input_length", shape=(1,), dtype="int64")
    label_length = layers.Input(name="label_length", shape=(1,), dtype="int64")

    # CNN Feature Extractor
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Reshape for RNN: from (batch, width, height, channels) to (batch, width, features)
    new_shape = ((IMG_WIDTH // 4), (IMG_HEIGHT // 4) * 64)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # RNN Layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(char_to_num.vocabulary_size() + 1, activation="softmax", name="output")(x)

    # CTC Loss Calculation
    output = CTCLayer(name="ctc_loss")(labels, x, input_length, label_length)

    # Define the final model
    model = keras.models.Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=output,
        name="cursive_reader_model"
    )

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam())
    return model

model = build_model()
model.summary()


### 3. TRAINING THE MODEL ###

epochs = 20  # You can increase this for better results
batch_size = 32

# Create data generators
train_generator = DataGenerator(train_samples, train_labels, batch_size)
validation_generator = DataGenerator(validation_samples, validation_labels, batch_size)

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model_checkpoint = keras.callbacks.ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint],
)

### 4. INFERENCE AND PREDICTION ###

# Get the prediction model (without the CTC loss layer)
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="output").output
)

def ctc_decode(y_pred, input_length, greedy=True):
    # Use tf.keras.backend.ctc_decode for decoding
    decoded = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=greedy)
    return decoded[0][0]

def predict_and_display(prediction_model, num_samples=5):
    # Get a few random validation samples
    sample_indices = np.random.choice(len(validation_samples), num_samples)
    
    for i in sample_indices:
        img_path = validation_samples[i]
        true_label = validation_labels[i]
        
        # Preprocess the image
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, 1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.transpose(img, perm=[1, 0, 2])
        img = tf.expand_dims(img, axis=0) # Add batch dimension

        # Make prediction
        pred = prediction_model.predict(img)
        pred_input_length = np.ones(pred.shape[0]) * pred.shape[1]
        
        # Decode the prediction
        decoded_pred = ctc_decode(pred, pred_input_length)
        decoded_pred = tf.strings.reduce_join(num_to_char(decoded_pred)).numpy().decode("utf-8")

        # Display results
        _, ax = plt.subplots()
        ax.imshow(tf.squeeze(img).numpy().T, cmap="gray")
        ax.set_title(f"True: '{true_label}'\nPred: '{decoded_pred}'")
        ax.axis("off")
        plt.show()

# Run prediction on a few samples
predict_and_display(prediction_model)