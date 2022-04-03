from typing import Dict

import json

import pandas as pd

from sklearn.model_selection import train_test_split

import keras
from keras.layers import TextVectorization, Embedding, Flatten, Dropout, Dense, Conv1D, MaxPooling1D, Bidirectional, LSTM
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy, Recall, Precision
from keras.callbacks import EarlyStopping

DATA = 'twitter_parsed_dataset.csv'

MAX_TOKENS = 10000
MAX_LENGTH = 250
EMBDED_DIM = 100

RUNS = 10
EPOCHS = 100

EXPERIMENTS = [
    {"architecture": "mlp", "layers": 2, "units": 1024},
    {"architecture": "mlp", "layers": 3, "units": 1024},
    {"architecture": "mlp", "layers": 2, "units": 1024, "dropout":0.2},
    {"architecture": "mlp", "layers": 3, "units": 1024, "dropout":0.2},
    {"architecture": "mlp", "layers": 2, "units": 1024, "dropout":0.5},
    {"architecture": "mlp", "layers": 3, "units": 1024, "dropout":0.5},
    {"architecture": "cnn", "layers": 2, "filters": 128},
    {"architecture": "cnn", "layers": 3, "filters": 128},
    {"architecture": "lstm", "units": 64},
    {"architecture": "lstm", "units": 128},
]


def read_and_transform(fname: str) -> pd.DataFrame:

    df = pd.read_csv(DATA)

    df.loc[df.Annotation == 'none', 'label'] = 0
    df.loc[df.Annotation != 'none', 'label'] = 1

    train, test = train_test_split(df, test_size=0.2)

    X_train = train['Text'].tolist()
    y_train = train['label'].tolist()

    X_test = test['Text'].tolist()
    y_test = test['label'].tolist()

    return X_train, X_test, y_train, y_test

def create_model(tokenize: TextVectorization, hp: Dict) -> keras.Model:
    model = keras.Sequential()
    model.add(tokenize)
    model.add(Embedding(input_dim = MAX_TOKENS+1, output_dim=EMBDED_DIM))

    if hp['architecture'] == 'mlp':
        model.add(Flatten())
        for layer in range(hp['layers']):
            if 'dropout' in hp:
                model.add(Dropout(hp['dropout']))
            model.add(Dense(units=hp['units'], activation='relu'))

    if hp['architecture'] == 'cnn':
        for layer in range(hp['layers']):
            model.add(Conv1D(filters=hp['filters'], kernel_size=4, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())

    if hp['architecture'] == 'lstm':
        model.add(Bidirectional(LSTM(hp['units'], activation='relu')))
        model.add(Flatten())

    model.add(Dense(units=1, activation='sigmoid'))

    metrics = [
        BinaryAccuracy(),
        Recall(),
        Precision()
    ]

    model.compile(loss=BinaryCrossentropy(),
        optimizer='adam',
        metrics=metrics)


    return model

X_train, X_test, y_train, y_test = read_and_transform(DATA)

vectorizer = TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode='int',
    output_sequence_length=MAX_LENGTH
)
vectorizer.adapt(X_train)

results = []

for experiment in EXPERIMENTS:

    run_history = []

    for x in range(RUNS):
        model = create_model(vectorizer, experiment)

        callbacks = [EarlyStopping(patience=1)]
        history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=EPOCHS, callbacks=callbacks)
        run_history.append(history.history)
    results.append((experiment, run_history))

with open("results.json", "w+") as f:
    json.dump(results, f)