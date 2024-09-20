import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l2
import os


def run_lstm_1(train_sets, 
               val_sets, 
               model_name, 
               neurons=64, 
               hidden_layers=2, 
               p_epochs=30,
               dropout=0.20,
               lr=0.001, 
               early_stopping_ptn=15,
               reduce_plateau_ptn=5,
               rp_lr=0.05,
               path="model_experiments",
               p_batch_size=512):
    """
    Builds, compiles, and trains an LSTM model for multi-class classification.

    Parameters:
    -----------
    train_sets : tf.data.Dataset
        Training dataset containing features and labels.
    val_sets : tf.data.Dataset
        Validation dataset used for evaluating model performance during training.
    model_name : str
        The name of the model, used for saving model checkpoints.
    neurons : int, optional (default=64)
        Number of neurons in the first LSTM layer. Subsequent layers will have a fraction of this number.
    hidden_layers : int, optional (default=2)
        Number of LSTM hidden layers in the model.
    p_epochs : int, optional (default=30)
        Number of training epochs.
    dropout : float, optional (default=0.20)
        Dropout rate for regularizing the model in each LSTM layer.
    lr : float, optional (default=0.001)
        Learning rate for the Adam optimizer.
    early_stopping_ptn : int, optional (default=15)
        Number of epochs with no improvement in validation loss before training stops early.
    reduce_plateau_ptn : int, optional (default=5)
        Number of epochs with no improvement before reducing the learning rate.
    rp_lr : float, optional (default=0.05)
        Factor by which the learning rate will be reduced when the plateau is detected.
    path : str, optional (default="model_experiments")
        Directory path for saving model checkpoints.
    p_batch_size : int, optional (default=512)
        Batch size used during training.

    Returns:
    --------
    history : History object
        Keras History object containing training and validation loss/accuracy over epochs.
    
    Functionality:
    --------------
    - Constructs an LSTM model with the specified number of neurons and hidden layers.
    - Applies dropout for regularization.
    - Uses sparse categorical crossentropy as the loss function for multi-class classification (3 classes).
    - Trains the model on the provided training dataset, using validation data for early stopping.
    - Reduces the learning rate if the validation loss plateaus for a set number of epochs.
    - Saves model checkpoints during training.
    """
    for features, labels in train_sets.take(1):
        features_input_shape = features.shape[1:]
        break
    
   
    model = Sequential(name=model_name)

    # Input layer
    model.add(LSTM(neurons, input_shape=(features_input_shape), activation="tanh", return_sequences=True, name="input_layer_0"))

    # hidden layers
    for i in range(0, hidden_layers):
        model.add(LSTM(int(neurons/2), activation="tanh", return_sequences=True, name=f"hidden_layer_{i}"))
        model.add(Dropout(dropout, name=f"dropout_layer_{i}"))

    # Last hidden layer
    model.add(LSTM(int(neurons/4), activation="tanh", return_sequences=False, name=f"last_hidden_layer"))
    model.add(Dropout(dropout, name="last_dropout_layer_"))
    
    # Output layer
    model.add(Dense(3, activation="softmax", name="output_layer"))
    
    checkpoint_callback = create_model_checkpoint(model.name, path)
    early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_ptn, verbose=1)
    reduce_plateau = ReduceLROnPlateau(monitor="val_loss", factor=rp_lr, patience=reduce_plateau_ptn, verbose=2)

    # Compiling the model
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=lr), metrics=["accuracy"])

    # Training the model
    history = model.fit(train_sets,
                       validation_data=val_sets, 
                       batch_size=p_batch_size, 
                       epochs=p_epochs,
                       verbose=0,
                       callbacks=[checkpoint_callback, early_stopping, reduce_plateau]
                      )
    return history



def run_lstm_2(train_sets, 
               val_sets, 
               model_name, 
               neurons=64, 
               hidden_layers=2, 
               p_epochs=30,
               dropout=0.20,
               lr=0.001, 
               early_stopping_ptn=15,
               reduce_plateau_ptn=5,
               rp_lr=0.5,
               path="model_experiments",
               p_batch_size=512,
               kr_l2=0.1):
    """
    Builds, compiles, and trains a bidirectional LSTM model with kernel regularization, batch normalization, 
    and dropout for multi-class classification.

    Parameters:
    -----------
    train_sets : tf.data.Dataset
        Training dataset containing features and labels.
    val_sets : tf.data.Dataset
        Validation dataset used for evaluating model performance during training.
    model_name : str
        The name of the model, used for saving model checkpoints.
    neurons : int, optional (default=64)
        Number of neurons in the first LSTM layer. Subsequent layers will have a fraction of this number.
    hidden_layers : int, optional (default=2)
        Number of bidirectional LSTM hidden layers in the model.
    p_epochs : int, optional (default=30)
        Number of training epochs.
    dropout : float, optional (default=0.20)
        Dropout rate for regularizing the model in each LSTM layer.
    lr : float, optional (default=0.001)
        Learning rate for the Adam optimizer.
    early_stopping_ptn : int, optional (default=15)
        Number of epochs with no improvement in validation loss before training stops early.
    reduce_plateau_ptn : int, optional (default=5)
        Number of epochs with no improvement before reducing the learning rate.
    rp_lr : float, optional (default=0.5)
        Factor by which the learning rate will be reduced when the plateau is detected.
    path : str, optional (default="model_experiments")
        Directory path for saving model checkpoints.
    p_batch_size : int, optional (default=512)
        Batch size used during training.
    kr_l2 : float, optional (default=0.1)
        L2 regularization parameter to control the magnitude of weights and prevent overfitting.

    Returns:
    --------
    history : History object
        Keras History object containing training and validation loss/accuracy over epochs.

    Functionality:
    --------------
    - Constructs a bidirectional LSTM model with kernel regularization applied to the layers.
    - Adds batch normalization after each LSTM layer to stabilize learning.
    - Uses sparse categorical crossentropy as the loss function for multi-class classification (3 classes).
    - Applies dropout to prevent overfitting.
    - Trains the model on the provided training dataset, using validation data for early stopping.
    - Reduces the learning rate if the validation loss plateaus for a set number of epochs.
    - Saves model checkpoints during training.
    """
    for features, labels in train_sets.take(1):
        features_input_shape = features.shape[1:]
        break
    
   
    model = Sequential(name=model_name)

    # Input layer
    model.add(Bidirectional(LSTM(neurons, activation="tanh", return_sequences=True, kernel_regularizer=l2(kr_l2)), 
                            input_shape=features_input_shape, name="input_layer_0"))
    model.add(BatchNormalization())

    # hidden layers
    for i in range(hidden_layers):
        model.add(Bidirectional(LSTM(int(neurons/2), activation="tanh", return_sequences=True, kernel_regularizer=l2(kr_l2)), name=f"hidden_layer_{i}"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout, name=f"dropout_layer_{i}"))    

    # Last hidden layer
    model.add(Bidirectional(LSTM(int(neurons/4), activation="tanh", return_sequences=False, kernel_regularizer=l2(kr_l2)), name="last_hidden_layer"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout, name="Last_dropout_layer"))

    # Output layer
    model.add(Dense(3, activation="softmax", name="output_layer"))
    
    checkpoint_callback = create_model_checkpoint(model.name, path)
    early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_ptn, verbose=1)
    reduce_plateau = ReduceLROnPlateau(monitor="val_loss", factor=rp_lr, patience=reduce_plateau_ptn, verbose=2, min_lr=1e-4)

    # Compiling the model
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=lr), metrics=["accuracy"])

    # Training the model
    history = model.fit(train_sets,
                        validation_data=val_sets,
                        batch_size=p_batch_size,
                        epochs=p_epochs,
                        verbose=0,
                        callbacks=[checkpoint_callback, early_stopping, reduce_plateau]
                        )
    return history