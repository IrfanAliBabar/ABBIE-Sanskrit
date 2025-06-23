import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Dropout, Concatenate, LayerNormalization, Bidirectional,Masking
from keras.layers import MultiHeadAttention
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
from data_preprocessing import prepare_data
import keras.callbacks as callbacks
import keras.backend as K
from keras.models import load_model
import pickle
import tensorflow as tf

class DataGenerator(Sequence):
    def __init__(self, data, char_to_index, max_seq_length, batch_size=64):
        self.data = data
        self.char_to_index = char_to_index
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.indices = np.arange(len(data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_indices]
        X, Y = self.__data_generation(batch_data)
        return X, Y

    def __data_generation(self, batch_data):
        X_batch = []
        Y_batch = []
        for entry in batch_data:
            word = entry[3]
            target = np.zeros(len(word))
            for i in range(entry[4], entry[5]):
                target[i] = 1
            X_batch.append([self.char_to_index[char] for char in word])
            Y_batch.append(target)
        X_batch = pad_sequences(X_batch, maxlen=self.max_seq_length, padding="post", value=self.char_to_index['$'])
        Y_batch = pad_sequences(Y_batch, maxlen=self.max_seq_length, padding="post", value=0.0)
        Y_batch = np.array(Y_batch).reshape(-1, self.max_seq_length, 1)
        return X_batch, Y_batch


# Main function to train and save the model
def train_and_save_model(train_data, model_save_path,test_data ):
    batch_size = 64
    epochs = 40
    latent_dim = 64
    embedding_dim = 16
    num_heads = 2 

    input_sequences = []
    output_sequences = []
    char_set = set()

    for entry in train_data:
        target = np.zeros(len(entry[3]))
        word = entry[3]
        input_sequences.append(word)
        for i in range(entry[4], entry[5]):
            target[i] = 1
        output_sequences.append(target)
        char_set.update(word)

    max_seq_length = max(len(seq) for seq in input_sequences)
    char_set.add('$')
    char_to_index = {char: idx for idx, char in enumerate(char_set)}
    num_chars = len(char_set)

    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=1)

    train_generator = DataGenerator(train_data, char_to_index, max_seq_length, batch_size=batch_size)
    val_generator = DataGenerator(val_data, char_to_index, max_seq_length, batch_size=batch_size)

    input_layer = Input(shape=(max_seq_length,))
    masking_layer = Masking(mask_value=0)(input_layer)
    embedding_layer = Embedding(input_dim=num_chars, output_dim=embedding_dim, input_length=max_seq_length)(masking_layer)
    left_context = Bidirectional(LSTM(latent_dim, return_sequences=True))(embedding_layer)
    left_context_d = Dropout(0.5)(left_context)
    right_context = Bidirectional(LSTM(latent_dim, return_sequences=True, go_backwards=True))(embedding_layer)
    right_context_d = Dropout(0.5)(right_context)
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=latent_dim)(left_context_d, right_context_d)
    merged_context = Concatenate()([left_context_d, attention, right_context_d])
    dropout_layer = Dropout(0.5)(merged_context)
    normalization_layer = LayerNormalization()(dropout_layer)
    dense_output = Dense(1, activation="sigmoid")(normalization_layer)


    model = Model(inputs=input_layer, outputs=dense_output)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='MSE', metrics=['accuracy'])

 
    model.summary()
    # Access the LSTM layers in the model
    for layer in model.layers:

        if isinstance(layer, LSTM) or isinstance(layer, Bidirectional):

            lstm_layer = layer if isinstance(layer, LSTM) else layer.layer

            print(f"LSTM Activation: {lstm_layer.activation}")
    #input("_")
            print(f"LSTM Recurrent Activation: {lstm_layer.recurrent_activation}")
    #early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    #learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=0.00001)
    #history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=[early_stopping])

    model.fit(train_generator, epochs=epochs, validation_data=val_generator)#, learning_rate_reduction

    model.save(model_save_path)



    pred_start_indices, accuracy = evaluate_predictions(data, model, char_to_idx, seq_length_limit):
    print('predicted_starts:', predicted_starts)
    print("accuracy:", accuracy)
    return model, char_to_idx, seq_length_limit

def evaluate_predictions(data, model, char_to_idx, seq_length_limit):
    pred_start_indices = []
    correct_count = 0
    window_size = 4

    for sample in data:
        word_to_predict = sample[3]
        word_indices = [char_to_idx.get(char, char_to_idx['$']) for char in word_to_predict]
        padded_indices = pad_sequences([word_indices], maxlen=seq_length_limit, padding="post", value=char_to_idx['$'])

        output_probs = model.predict(padded_indices).reshape((seq_length_limit))

        max_value = 0
        start_index = 0
        for idx in range(seq_length_limit - window_size):
            window_sum = np.sum(output_probs[idx:idx + window_size])
            if window_sum > max_value:
                max_value = window_sum
                start_index = idx

        pred_start_indices.append(start_index)
        true_start_index = sample[4]

        if start_index == true_start_index:
            correct_count += 1

    accuracy = correct_count / len(data)
    return pred_start_indices, accuracy

if __name__ == "__main__":
    dataset_path = 'C:/Users/....../Sanskrit_dataset.csv'
    model_save_path = 'loc_split.h5'

    data = prepare_data(dataset_path)
    print(f'Total samples in dataset: {len(data)}')

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)
    print(f'Training samples: {len(train_data)}')

    model, char_to_index, max_seq_length = train_and_save_model(train_data, model_save_path,test_data)
    # Save char_to_index and max_seq_length
    with open('char_to_index.pkl', 'wb') as f:
        pickle.dump(char_to_index, f)

    with open('max_seq_length.pkl', 'wb') as f:
        pickle.dump(max_seq_length, f)
