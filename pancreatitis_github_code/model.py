import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

class DeepSurvModel:
    def __init__(self, input_dim):
        self.model = self.build_model(input_dim)

    def build_model(self, input_dim):
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1)(x)
        model = models.Model(inputs, output)
        model.compile(optimizer='adam', loss=self.cox_loss)
        return model

    def cox_loss(self, y_true, y_pred):
        event = y_true[:, 0]  # event indicator
        risk = y_pred[:, 0]   # predicted log-risk
        time = y_true[:, 1]   # survival time

        order = tf.argsort(time, direction='DESCENDING')
        sorted_event = tf.gather(event, order)
        sorted_risk = tf.gather(risk, order)

        hazard_ratio = tf.math.exp(sorted_risk)
        log_risk = tf.math.log(tf.math.cumsum(hazard_ratio))
        uncensored_likelihood = sorted_risk - log_risk

        censored_likelihood = tf.multiply(uncensored_likelihood, sorted_event)
        neg_loss = -K.mean(censored_likelihood)
        return neg_loss

    def train(self, X, time, event, epochs=100):
        y_train = tf.convert_to_tensor(tf.stack([event, time], axis=1), dtype=tf.float32)
        self.model.fit(X, y_train, epochs=epochs, batch_size=32, verbose=1)

    def predict(self, X):
        return self.model.predict(X).flatten()