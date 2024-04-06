import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Make sure you have CUDA available on your machine
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the Kernel Filter Layer
class KernelFilter(nn.Module):
    def __init__(self, input_dim, kernel_size, num_kernels):
        super(KernelFilter, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=num_kernels, kernel_size=kernel_size)
    def forward(self, x):
        # The input x should already be in the shape (batch_size, input_dim, time_series_length)
        return self.conv(x)
# Define the Time Attention Layer
class TimeAttention(nn.Module):
    def __init__(self, feature_dim):
        super(TimeAttention, self).__init__()
        self.attention = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, x):
        # x shape: (batch_size, time_series_length, feature_dim)
        attention_weights = torch.softmax(self.attention(x), dim=1)
        attended_features = attention_weights * x # Apply attention weights
        return attended_features.sum(dim=1).unsqueeze(1)  # Sum along the time_series_length dimension

# Define the TNN Model
class TimeSeriesNeuralNetwork(nn.Module):
    def __init__(self, input_dim, kernel_size, num_kernels, output_dim):
        super(TimeSeriesNeuralNetwork, self).__init__()
        self.kernel_filter = KernelFilter(input_dim, kernel_size, num_kernels)
        self.time_attention = TimeAttention(num_kernels)
        self.output_layer = nn.Linear(1, output_dim)
    
    def forward(self, x):
        # x should already be of shape (batch_size, time_series_length, input_dim)
        # You need to permute it to (batch_size, input_dim, time_series_length) to match Conv1d input requirements
        x = x.permute(0, 2, 1)
        features = self.kernel_filter(x).squeeze(2)# Since kernel size is 2 and sequence length is 2, L will be 1
        # Now features should be of shape (batch_size, num_kernels*output_channels, L)
        # Where L is the resulting sequence length from the convolutions
        attention_output = self.time_attention(features)
        # Assuming you still want to use only the last time step for prediction
        return self.output_layer(attention_output)
    
# Hyperparameters
input_dim = 5 # This should match your time-series data input dimensions
kernel_size = 2  # Example kernel size
num_kernels = 2  # Example number of kernels
output_dim = 1  # Output dimension for prediction

# Initialize the model
model = TimeSeriesNeuralNetwork(input_dim=5, kernel_size=2, num_kernels=2, output_dim=1).to(device)
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
columns_to_drop = ['Date', 'Dividends', 'Stock Splits']
df = df.drop(columns=columns_to_drop)

# Preprocess data: scale values and create sequences if needed
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']].values)


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
model = TimeSeriesNeuralNetwork(input_dim=5, kernel_size=2, num_kernels=2, output_dim=1).to(device)
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
