import torch
import torch.nn as nn
import torch.optim as optim

# Define the Kernel Filter Layer
class KernelFilter(nn.Module):
    def __init__(self, input_dim, kernel_size, num_kernels):
        super(KernelFilter, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=input_dim, out_channels=1, kernel_size=kernel_size) for _ in range(num_kernels)])
    
    def forward(self, x):
        # x shape: (batch_size, time_series_length, input_dim)
        x = x.permute(0, 2, 1)  # Switch to (batch_size, input_dim, time_series_length)
        outputs = [conv(x).squeeze(1) for conv in self.convs]
        return torch.cat(outputs, dim=1)  # Concatenate along the feature dimension

# Define the Time Attention Layer
class TimeAttention(nn.Module):
    def __init__(self, feature_dim):
        super(TimeAttention, self).__init__()
        self.attention = nn.Linear(feature_dim, 1)
    
    def forward(self, x):
        # x shape: (batch_size, time_series_length, feature_dim)
        attention_weights = torch.softmax(self.attention(x), dim=1)
        return torch.sum(attention_weights * x, dim=1)  # Weighted sum of features over time

# Define the TNN Model
class TimeSeriesNeuralNetwork(nn.Module):
    def __init__(self, input_dim, kernel_size, num_kernels, output_dim):
        super(TimeSeriesNeuralNetwork, self).__init__()
        self.kernel_filter = KernelFilter(input_dim, kernel_size, num_kernels)
        self.time_attention = TimeAttention(num_kernels)
        self.output_layer = nn.Linear(num_kernels, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, time_series_length, input_dim)
        features = self.kernel_filter(x)
        features = features.permute(0, 2, 1)  # Switch to (batch_size, time_series_length, num_kernels)
        attention_output = self.time_attention(features)
        return self.output_layer(attention_output)

# Hyperparameters
input_dim = 10  # This should match your time-series data input dimensions
kernel_size = 3  # Example kernel size
num_kernels = 5  # Example number of kernels
output_dim = 1  # Output dimension for prediction

# Initialize the model
model = TimeSeriesNeuralNetwork(input_dim, kernel_size, num_kernels, output_dim)
print(model)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example dummy data for illustration (you should replace this with your actual dataset)
dummy_inputs = torch.randn(32, 100, input_dim)  # batch_size, time_series_length, input_dim
dummy_outputs = torch.randn(32, output_dim)  # batch_size, output_dim

# Training loop (simplified for illustration)
for epoch in range(10):  # Let's say 10 epochs for now
    optimizer.zero_grad()
    predictions = model(dummy_inputs)
    loss = criterion(predictions, dummy_outputs)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
