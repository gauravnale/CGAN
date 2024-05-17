import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class CGAN:
    def __init__(self, input_dim, num_classes, gen_hidden_dim, disc_hidden_dim):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.gen_hidden_dim = gen_hidden_dim
        self.disc_hidden_dim = disc_hidden_dim
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.combined = self.build_combined()

        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        
        # Compile combined model
        self.combined.compile(loss='binary_crossentropy', optimizer=Adam())
        
    def build_generator(self):
        input_noise = tf.keras.Input(shape=(self.input_dim,))
        input_label = tf.keras.Input(shape=(1,))
        x = Concatenate()([input_noise, input_label])
        x = Dense(self.gen_hidden_dim, activation='relu')(x)
        x = Dense(self.gen_hidden_dim, activation='relu')(x)
        output = Dense(self.input_dim, activation='tanh')(x)  # Use 'tanh' to constrain output between -1 and 1
        return Model(inputs=[input_noise, input_label], outputs=output, name='generator')
    
    def build_discriminator(self):
        input_data = tf.keras.Input(shape=(self.input_dim,))
        input_label = tf.keras.Input(shape=(1,))
        x = Concatenate()([input_data, input_label])
        x = Dense(self.disc_hidden_dim, activation='relu')(x)
        x = Dense(self.disc_hidden_dim, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        return Model(inputs=[input_data, input_label], outputs=output, name='discriminator')
    
    def build_combined(self):
        noise = tf.keras.Input(shape=(self.input_dim,))
        label = tf.keras.Input(shape=(1,))
        generated_data = self.generator([noise, label])
        self.discriminator.trainable = False
        validity = self.discriminator([generated_data, label])
        return Model(inputs=[noise, label], outputs=validity)
    
    def train(self, X_train, y_train, epochs, batch_size=128, sample_interval=50):
        X_train = np.array(X_train)
        y_train = np.array(y_train).reshape(-1, 1)
        
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_data, labels = X_train[idx], y_train[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            gen_data = self.generator.predict([noise, labels])
            
            d_loss_real = self.discriminator.train_on_batch([real_data, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_data, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)
            
            if epoch % sample_interval == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Load the dataset
df = pd.read_excel("/Users/sahilwikhe/Familogy (2024 intern ) /CGAN/FamilyOfficeEntityDataSampleV1.1.xlsx", sheet_name= "Financial Assets")

# Preprocessing
label_encoders = {}
def preprocess_data(df):
    # Encode categorical variables
    cat_cols = ['Asset Type', 'Asset Details']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Remove '%' symbols and convert to float
    df["Ownership Percentage"] = df['Ownership Percentage'].replace("[\%]", "", regex=True).astype(float)
    df['Return on Investments'] = pd.to_numeric(df['Return on Investments'].replace("[\%]", "", regex=True), errors='coerce')
    df["Return on Investments"].fillna(0, inplace=True)
    df["Value"] = df["Value"].replace("[\$,]", "", regex=True).replace("-", np.nan).astype(float)
    df["Ownership Percentage"].fillna(0, inplace=True)

    # Drop unnecessary columns
    df.drop(['AssetID', 'Asset_Date', 'ClientID', 'EVaultID'], axis=1, inplace=True)

    return df

# Preprocess the data
processed_data = preprocess_data(df)

# Define input dimensions and number of classes
input_dim = processed_data.shape[1] - 1  # Excluding the target variable (Status)
num_classes = 2  # Assuming two classes for 'Status' (e.g., Client.Active, Client.Inactive)

# Scale the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
processed_data_scaled = scaler.fit_transform(processed_data)

# Define CGAN parameters
gen_hidden_dim = 128
disc_hidden_dim = 128
epochs = 10000
batch_size = 32

# Instantiate CGAN
cgan = CGAN(input_dim, num_classes, gen_hidden_dim, disc_hidden_dim)

# Train CGAN
cgan.train(processed_data_scaled[:, :-1], processed_data_scaled[:, -1], epochs=epochs, batch_size=batch_size)

def generate_synthetic_samples(cgan, num_samples):
    noise = np.random.normal(0, 1, (num_samples, cgan.input_dim))
    sampled_labels = np.random.randint(0, cgan.num_classes, num_samples).reshape(-1, 1)
    generated_data = cgan.generator.predict([noise, sampled_labels])
    return generated_data

# Generate 10 synthetic samples
num_samples = 10
synthetic_samples = generate_synthetic_samples(cgan, num_samples)

# Inverse transform the synthetic data to original scale
synthetic_samples_original_scale = scaler.inverse_transform(np.hstack((synthetic_samples, np.zeros((num_samples, 1)))))

# Decode the categorical features back to their original labels
synthetic_df = pd.DataFrame(data=synthetic_samples_original_scale, columns=processed_data.columns)
for col in ['Asset Type', 'Asset Details']:
    le = label_encoders[col]
    synthetic_df[col] = le.inverse_transform(synthetic_df[col].astype(int))

# Print synthetic data
print("Synthetic Data:")
print(synthetic_df)
