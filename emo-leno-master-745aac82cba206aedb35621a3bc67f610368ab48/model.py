from keras import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Embedding, BatchNormalization, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, Dropout, Dense

import config
import data

weights, X_train, y_train, X_test, y_test = data.load_data(config)

callbacks = [
    EarlyStopping(monitor='loss', min_delta=1e-4, patience=6, verbose=1),
    ReduceLROnPlateau(monitor='loss', factor=0.1, epsilon=0.0001, patience=2, cooldown=1, verbose=1),
    ModelCheckpoint(filepath='./checkpoints/weights.epoch_{epoch:02d}-val_acc_{val_acc:.2f}.h5', monitor='loss',
                    verbose=0, save_best_only=True),
]

CNNBranch = Sequential()
CNNBranch.add(Embedding(len(weights),
                        output_dim=config.dims,
                        weights=[weights],
                        input_length=config.sequence_length)
              )
CNNBranch.add(BatchNormalization())
CNNBranch.add(SpatialDropout1D(rate=config.dropout))
CNNBranch.add(Conv1D(filters=config.nb_filter,
                     kernel_size=config.filter_length,
                     padding='valid',
                     activation='relu',
                     strides=1))
CNNBranch.add(GlobalMaxPooling1D())
CNNBranch.add(BatchNormalization())
CNNBranch.add(Dropout(config.dropout))
CNNBranch.add(Dense(4, activation='softmax'))
CNNBranch.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
CNNBranch.summary()
CNNBranch.fit(X_train,
              y_train,
              batch_size=config.batch_size,
              epochs=config.nb_epoch,
              callbacks=callbacks,
              validation_data=(X_test, y_test))
