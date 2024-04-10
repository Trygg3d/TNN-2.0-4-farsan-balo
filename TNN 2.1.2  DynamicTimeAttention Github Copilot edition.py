import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader 
from torch.utils.checkpoint import checkpoint
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

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
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


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


class DynamicTimeAttention(nn.Module):
    def __init__(self, input_dim, num_kernels, kernel_size, feature_dim, num_heads):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be a multiple of num_heads"
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.kernel_filters = nn.Conv1d(input_dim, num_kernels, kernel_size, padding=kernel_size // 2)
        self.query = nn.Linear(num_kernels, feature_dim)
        self.key = nn.Linear(num_kernels, feature_dim)
        self.value = nn.Linear(num_kernels, feature_dim)
        self.adaptive_weight_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_heads),
            nn.Softmax(dim=-1)
        )
        self.fc_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, mask=None):
        #print(f"Input shape: {x.shape}")  # Print the shape of the input tensor
        x = self.kernel_filters(x.permute(0, 2, 1)).permute(0, 2, 1)
        #print(f"Shape after kernel_filters: {x.shape}")  # Print the shape of the tensor after the kernel_filters operation
        batch_size, sequence_length, _ = x.size()
        query = self.query(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        #print(f"Query shape: {query.shape}")  # Print the shape of the query tensor
        key = self.key(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.scale), dim=-1)
        weighted = torch.einsum("nhql,nlhd->nqhd", [attention, value])
        weighted = weighted.transpose(1, 2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)
        #print(f"Output shape: {weighted.shape}")  # Print the shape of the output tensor
        return self.fc_out(weighted)


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout_rate, forward_expansion, kernel_size):
        super(TransformerBlock, self).__init__()
        self.attention = DynamicTimeAttention(input_dim, kernel_size, kernel_size, input_dim, num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, forward_expansion * input_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * input_dim, input_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        #print(f"Input shape: {x.shape}")  # Print the shape of the input tensor
        attention = self.attention(x, mask)
        #print(f"Attention output shape: {attention.shape}")  # Print the shape of the output tensor
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        return self.dropout(self.norm2(forward + x))


class TimeSeriesNeuralNetwork(nn.Module):
    def __init__(self, input_dim, kernel_size, num_kernels, output_dim, num_heads, max_len=5000):
        super(TimeSeriesNeuralNetwork, self).__init__()
        self.num_filters = 8
        self.positional_encoding = PositionalEncoding(input_dim, max_len)
        self.kernel_filters = nn.ModuleList([
            nn.Conv1d(input_dim, num_kernels, kernel_size, padding=kernel_size // 2) 
            for _ in range(self.num_filters)
        ])
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(num_kernels * self.num_filters, num_heads, 0.1, 512, kernel_size)
            for _ in range(2)
        ])
        self.fc_out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x + self.positional_encoding(x)
        x = x.permute(0, 2, 1)
        x = torch.cat([filter(x) for filter in self.kernel_filters], dim=1)
        x = x.permute(0, 2, 1)
        # Use checkpointing for the transformer blocks
        for transformer in self.transformer_blocks:
            x = checkpoint(transformer, x, None, use_reentrant=False)
        
        return self.fc_out(x)





# Hyperparameters
input_dim = 4
kernel_size = 1
num_kernels = 8
output_dim = 4
num_heads = 8
feature_dim = 4
head_dim = feature_dim // num_heads
learning_rate = 0.001
step_size = 50
gamma = 0.1
batch_size = 512 # Adjust based on your system's memory


# Data loading and preprocessing
df = pd.read_csv(r'C:\Users\andru\OneDrive\Skrivbord\EURUSD OHLC data m5\EURUSD_M1_201201020000_202312292356.csv', sep='\t')
print("Data loaded")

# Drop unnecessary columns
columns_to_drop = ['<DATE>','<TIME>','<TICKVOL>','<VOL>','<SPREAD>']
df = df.drop(columns=columns_to_drop)
print("Columns dropped")
# Assuming 'df' is your DataFrame with the columns ['Open', 'High', 'Low', 'Close']

# Missing Value Treatment: Linear Interpolation
df = df.interpolate(method='linear', axis=0).bfill().ffill()
print("Missing values treated")

# Calculate the mean and standard deviation for each feature
mean = df.mean()
std_dev = df.std()

# Define the upper and lower bounds for normal data (within 3σ)
lower_bound = mean - 3 * std_dev
upper_bound = mean + 3 * std_dev

# Replace outliers with the median of the corresponding feature
for column in ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']:
    median_value = df[column].median()
    df[column] = np.where(df[column] > upper_bound[column], median_value, df[column])
    df[column] = np.where(df[column] < lower_bound[column], median_value, df[column])

# Continue with your preprocessing steps...

# Preprocess data: scale values and create sequences if needed
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(df[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].values)


# Convert to PyTorch tensors
tensor_data = torch.tensor(scaled_data, dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def create_sequences(input_data, sequence_length):
    sequences = []
    labels  = []
    for i in range(len(input_data) - 2*sequence_length): 
        sequences.append(input_data[i:i+sequence_length])
        labels.append(input_data[i+sequence_length:i+2*sequence_length])  

    return torch.stack(sequences).to(device), torch.stack(labels).to(device)

# Example: Create sequences of 60 minutes
sequence_length = 1
sequential_data, sequential_labels = create_sequences(tensor_data, sequence_length)

# First, split the data into training (70%) and temporary (30%) sets
train_data, temp_data, train_labels, temp_labels = train_test_split(sequential_data, sequential_labels, test_size=0.3, random_state=42, shuffle=True)

# Then, split the temporary data into validation (15%) and test (15%) sets
val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42, shuffle=True)

# Create DataLoader for batch processing
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)
test_dataset = TensorDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=0)  # Shuffling is usually not done for time series
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)

# Set up the model, loss function, and optimizer
for lr in [0.001, 0.01, 0.1]:
    model = TimeSeriesNeuralNetwork(input_dim=4, kernel_size=1, num_kernels=8, output_dim=4, num_heads=num_heads).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)


    # Training loop
    model.train()
    for epoch in range(10):
        print(f"Epoch {epoch+1}/10")
        total_loss = 0
        progress_bar = tqdm(total=len(train_loader), desc='Training', position=0)
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(sequences)
            ##print(f'Labels shape: {labels.shape}')  # Add this line
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
            
            # Update progress bar
            progress_bar.update(1)

        progress_bar.close()
        train_error = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')


        # Validation stage
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():  # Disable gradient computation
            for batch in val_loader:
                inputs, targets = batch
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        val_error = total_val_loss / len(val_loader)
        print("Validation Loss:", total_val_loss / len(val_loader))

        # Test stage
        model.eval()  # Set the model to evaluation mode
        total_test_loss = 0
        with torch.no_grad():  # Disable gradient computation
            for batch in test_loader:
                inputs, targets = batch
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_test_loss += loss.item()
        print("Test Loss:", total_test_loss / len(test_loader))

# Print the final training and validation errors for this learning rate
print(f"Learning rate: {lr}, Training error: {train_error}, Validation error: {val_error}")
