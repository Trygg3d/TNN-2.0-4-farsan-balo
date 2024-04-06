import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split

# Make sure you have CUDA available on your machine
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.scale = 1 / math.sqrt(feature_dim)

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, value)

class TimeAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        self.attention = ScaledDotProductAttention(self.head_dim)

        self.fc_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_length, _ = query.size()

        # Linear projections for multi-head attention
        query = self.query(query).view(batch_size, seq_length, self.num_heads, self.head_dim)
        key = self.key(key).view(batch_size, seq_length, self.num_heads, self.head_dim)
        value = self.value(value).view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Transpose for attention dot product: b x h x s x d
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Same mask applied to all heads.
            attention = self.attention(query, key, value, mask)
        else:
            attention = self.attention(query, key, value)

        # Concatenate attention heads and put through final linear layer
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_length, self.feature_dim)
        out = self.fc_out(attention)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout_rate, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = TimeAttention(input_dim, num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, forward_expansion * input_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * input_dim, input_dim)
        )
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        # Apply attention mechanism
        attention = self.attention(x, x, x, mask)
        # Apply dropout and residual connection after attention
        x = self.dropout(self.norm1(attention + x))
        
        forward = self.feed_forward(x)
        # Apply dropout and residual connection after feed forward
        out = self.dropout(self.norm2(forward + x))
        return out

# Kernel Filter Layer
class KernelFilter(nn.Module):
    def __init__(self, input_dim, kernel_size, num_kernels):
        super(KernelFilter, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=num_kernels, kernel_size=kernel_size)

    def forward(self, x):
        return F.relu(self.conv(x))



# TimeSeries Neural Network
class TimeSeriesNeuralNetwork(nn.Module):
    def __init__(self, input_dim, kernel_size, num_kernels, output_dim, num_heads, max_len=5000):
        super(TimeSeriesNeuralNetwork, self).__init__()
        self.num_filters = 8  # Define the number of parallel filters

        # Embedding layers: positional encoding layer might be used after this embedding
        self.positional_encoding = PositionalEncoding(d_model=input_dim, max_len=max_len)

        # Create parallel kernel filters
        self.kernel_filters = nn.ModuleList([
            KernelFilter(input_dim=input_dim, kernel_size=kernel_size, num_kernels=num_kernels) 
            for _ in range(self.num_filters)
        ])

        # Create a Transformer block
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(input_dim=num_kernels, num_heads=num_heads, dropout_rate=0.1, forward_expansion=4)
            for _ in range(self.num_filters)
        ])

        # Linear output layer
        self.output_layer = nn.Linear(self.num_filters * num_kernels, output_dim)
    
    def forward(self, x):
        # Apply positional encoding
        x = self.positional_encoding(x)

        # x shape should be (batch_size, seq_length, input_dim)
        # Apply kernel filters and transformer blocks
        filter_outputs = []
        for kernel_filter, transformer_block in zip(self.kernel_filters, self.transformer_blocks):
            # Apply kernel filter
            filtered_x = kernel_filter(x.permute(0, 2, 1))
            filtered_x = filtered_x.permute(0, 2, 1)

            # Apply Transformer block
            transformed_x = transformer_block(filtered_x, mask=None)
            filter_outputs.append(transformed_x)

        # Concatenate the output of each filter to form the final feature set
        concatenated_features = torch.cat(filter_outputs, dim=2)

        # Flatten and apply final linear layer for predictions
        final_features = concatenated_features.view(concatenated_features.size(0), -1)
        return self.output_layer(final_features)


# Hyperparameters
input_dim = 4 # This should match your time-series data input dimensions
kernel_size = 2  # Example kernel size
num_kernels = 2  # Example number of kernels
output_dim = 1  # Output dimension for prediction
num_heads = 2  # Example: number of attention heads

# Initialize the model
model = TimeSeriesNeuralNetwork(input_dim=4, kernel_size=2, num_kernels=2, output_dim=1, num_heads=num_heads).to(device)
print(model)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example dummy data for illustration (you should replace this with your actual dataset)
dummy_inputs = torch.randn(32, 100, input_dim)  # batch_size, time_series_length, input_dim
dummy_outputs = torch.randn(32, output_dim)  # batch_size, output_dim

# Data loading and preprocessing
df = pd.read_csv(r'C:\Users\andru\OneDrive\Skrivbord\S&P 500 yfinance data ohlc 1 day\SP500_data.csv', sep=',')

# Drop unnecessary columns
columns_to_drop = ['Date','Volume','Dividends','Stock Splits']
df = df.drop(columns=columns_to_drop)

# Assuming 'df' is your DataFrame with the columns ['Open', 'High', 'Low', 'Close']

# Missing Value Treatment: Linear Interpolation
df = df.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')

# Calculate the mean and standard deviation for each feature
mean = df.mean()
std_dev = df.std()

# Define the upper and lower bounds for normal data (within 3σ)
lower_bound = mean - 3 * std_dev
upper_bound = mean + 3 * std_dev

# Replace outliers with the median of the corresponding feature
for column in ['Open', 'High', 'Low', 'Close']:
    median_value = df[column].median()
    df[column] = np.where(df[column] > upper_bound[column], median_value, df[column])
    df[column] = np.where(df[column] < lower_bound[column], median_value, df[column])

# Continue with your preprocessing steps...

# Preprocess data: scale values and create sequences if needed
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close']].values)


# Convert to PyTorch tensors
tensor_data = torch.tensor(scaled_data, dtype=torch.float32)

# Define a function to create sequences
def create_sequences(input_data, sequence_length):
    sequences = []
    labels  = []
    for i in range(len(input_data) - sequence_length):
        sequences.append(input_data[i:i+sequence_length])
        labels.append(input_data[i + sequence_length])

    return torch.stack(sequences), torch.stack(labels)

# Example: Create sequences of 1 days
sequence_length = 2
sequential_data, sequential_labels = create_sequences(tensor_data, sequence_length)

# Split data into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(sequential_data, sequential_labels, test_size=0.2, shuffle=False)

# Create DataLoader for batch processing
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Shuffling is usually not done for time series

# Set up the model, loss function, and optimizer
model = TimeSeriesNeuralNetwork(input_dim=4, kernel_size=2, num_kernels=2, output_dim=1, num_heads=num_heads).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(10):  # example epoch count
    total_loss = 0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

