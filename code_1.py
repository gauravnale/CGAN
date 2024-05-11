import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class CGAN:
    def __init__(self, input_dim, num_classes, gen_hidden_dim, disc_hidden_dim):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.gen_hidden_dim = gen_hidden_dim
        self.disc_hidden_dim = disc_hidden_dim
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.combined = self.build_combined()

        self.generator.compile(loss='binary_crossentropy', optimizer=Adam())

        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        
        # Compile combined model
        self.combined.compile(loss='binary_crossentropy', optimizer=Adam())
        
    def build_generator(self):
        input_noise = tf.keras.Input(shape=(self.input_dim,))
        input_label = tf.keras.Input(shape=(1,))
        x = Concatenate()([input_noise, input_label])
        x = Dense(self.gen_hidden_dim, activation='relu')(x)
        x = Dense(self.gen_hidden_dim, activation='relu')(x)
        output = Dense(self.input_dim, activation='linear')(x)
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

df = pd.read_excel("FamilyOfficeEntityDataSampleV1.1.xlsx")  # Replace "your_file.xlsx" with the path to your Excel file

# Preprocessing
def preprocess_data(df):
    # Convert Date of Birth to age
    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'])
    # df['Age'] = (pd.to_datetime('now') - df['Date of Birth']).astype('<m8[Y]')
    df.drop('Date of Birth', axis=1, inplace=True)

    # Encode categorical variables
    cat_cols = ['Marital Status', 'State', 'Profession', 'Financial Goals', 'Status']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # Convert Net Worth to numerical
    df['Net Worth'] = df['Net Worth'].replace('[\$,]', '', regex=True).astype(float)

    # Drop unnecessary columns
    df.drop(['ClientID', 'First Name', 'Last Name', 'Contact Information', 'Address'], axis=1, inplace=True)
    
    return df

# Preprocess the data
processed_data = preprocess_data(df)

# Define input dimensions and number of classes
input_dim = processed_data.shape[1] - 1  # Excluding the target variable (Status)
num_classes = 2  # Assuming two classes for 'Status' (e.g., Client.Active, Client.Inactive)

# Scale the data
scaler = StandardScaler()
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

def convert_numerical_to_words(numerical_output):
    # Map numerical values to words or descriptions
    marital_status_map = {0: "Single", 1: "Married", 2: "Divorced", 3: "Widowed"}
    status_map = {0: "Client.Inactive", 1: "Client.Active"}

    # Convert numerical values to words
    marital_status_word = marital_status_map.get(numerical_output["Marital Status"], "Unknown")
    status_word = status_map.get(numerical_output["Status"], "Unknown")

    # Format numerical values
    dependents = str(numerical_output["Dependents"])
    state = numerical_output["State"]
    profession = numerical_output["Profession"]
    net_worth = "${:,.2f}".format(numerical_output["Net Worth"])
    financial_goals = numerical_output["Financial Goals"]

    return {
        "Marital Status": marital_status_word,
        "Dependents": dependents,
        "State": state,
        "Profession": profession,
        "Net Worth": net_worth,
        "Financial Goals": financial_goals,
        "Status": status_word
    }

synthetic_data_words = []
for synthetic_sample in synthetic_samples:
    numerical_output = {
        "Marital Status": synthetic_sample[0],     
        "Dependents": synthetic_sample[1],         
        "State": "North Carolina",                 
        "Profession": "Entrepreneur",              
        "Net Worth": synthetic_sample[2],          
        "Financial Goals": "Save for children's education, retire at 60",  
        "Status": 1                                
    }
    synthetic_data_words.append(convert_numerical_to_words(numerical_output))

# Create DataFrame for synthetic data
synthetic_df = pd.DataFrame(data=[synthetic_data_words])

# Save synthetic data to Excel
synthetic_df.to_excel("synthetic_data_words.xlsx", index=False)