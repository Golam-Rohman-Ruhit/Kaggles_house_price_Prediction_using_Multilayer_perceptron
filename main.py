import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# --- 1. SET RANDOM SEED (For 100% Reproducibility) ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# --- 2. LOAD DATA ---
print("Loading datasets...")
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, 'train.csv')
test_path = os.path.join(script_dir, 'test.csv')

try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Files loaded successfully!")
except FileNotFoundError:
    print("Error: 'train.csv' and 'test.csv' not found in project folder!")
    exit()

test_ids = test_df['Id'].copy()

# --- 3. DATA PREPROCESSING (Advanced Techniques for Max Accuracy) ---
# Separate Target
y = train_df['SalePrice'].values
train_df.drop(['SalePrice', 'Id'], axis=1, inplace=True)
test_df.drop(['Id'], axis=1, inplace=True)

# Combine for processing
all_data = pd.concat([train_df, test_df], axis=0)

# **TRICK 1: MSSubClass is actually categorical, not numerical**
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

# Handle Missing Values
for col in all_data.columns:
    if all_data[col].dtype == 'object':
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    else:
        all_data[col] = all_data[col].fillna(all_data[col].median())

# **TRICK 2: Fix Skewness in Input Features (The Secret to High Score)**
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: x.skew()).sort_values(ascending=False)
high_skew = skewed_feats[skewed_feats > 0.75]
skew_index = high_skew.index
# Apply Log1p to skewed features
all_data[skew_index] = np.log1p(all_data[skew_index])
print(f"Applied log transform to {len(skew_index)} skewed input features.")

# One-Hot Encoding
all_data = pd.get_dummies(all_data)
print(f"Total features after One-Hot Encoding: {all_data.shape[1]}")

# Split back to Train and Test
X = all_data.iloc[:len(train_df)].values
X_test_final = all_data.iloc[len(train_df):].values

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test_final = scaler.transform(X_test_final)

# Log Transformation of Target
y_log = np.log1p(y)
y_tensor = y_log.reshape(-1, 1)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y_tensor, test_size=0.2, random_state=SEED)

# Tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test_final)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# --- 4. MODEL ARCHITECTURE (Deep & Wide with Batch Norm) ---
class HousePriceMLP(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceMLP, self).__init__()
        self.model = nn.Sequential(
            # Layer 1: Input -> 512 (Wider layer)
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),  # Batch Norm stabilizes training
            nn.ReLU(),
            nn.Dropout(0.4),  # Slightly higher dropout

            # Layer 2: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Output Layer: 128 -> 1
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


model = HousePriceMLP(X.shape[1])
print("\n--- Model Architecture ---")
print(model)

# --- 5. TRAINING ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# StepLR to reduce learning rate gradually
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
epochs = 600  # Increased epochs for perfect convergence

train_losses = []
val_losses = []

print("\nStarting Training...")
for epoch in range(epochs):
    model.train()
    batch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()

    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor)
        val_loss = criterion(val_preds, y_val_tensor)

    train_losses.append(batch_loss / len(train_loader))
    val_losses.append(val_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}')

# --- 6. VISUALIZATIONS ---

# Graph 1: Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.title('Training and Validation Loss (Optimized)')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
print("Graph 1 Saved: loss_curve.png")

# Graph 2: Prediction Scatter
model.eval()
with torch.no_grad():
    val_preds_log = model(X_val_tensor).numpy()
    val_actual_log = y_val_tensor.numpy()

val_preds_real = np.expm1(val_preds_log)
val_actual_real = np.expm1(val_actual_log)

plt.figure(figsize=(8, 8))
plt.scatter(val_actual_real, val_preds_real, alpha=0.5, color='blue')
plt.plot([val_actual_real.min(), val_actual_real.max()], [val_actual_real.min(), val_actual_real.max()], 'r--', lw=2)
plt.title('Validation: Predictions vs Actual (High Accuracy)')
plt.xlabel('Actual Prices ($)')
plt.ylabel('Predicted Prices ($)')
plt.grid(True)
plt.savefig('prediction_scatter.png')
print("Graph 2 Saved: prediction_scatter.png")

# Generate Test Predictions
model.eval()
with torch.no_grad():
    test_preds_log = model(X_test_tensor).numpy()
    test_preds_real = np.expm1(test_preds_log).flatten()

# Graph 3: Distribution
plt.figure(figsize=(10, 6))
plt.hist(test_preds_real, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Predicted House Price ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Test Predictions')
plt.grid(True, alpha=0.3)
plt.savefig('histogram_distribution.png')
print("Graph 3 Saved: histogram_distribution.png")

# Graph 4: Comparison
plt.figure(figsize=(10, 6))
y_train_real = np.expm1(y_log)
plt.hist(y_train_real, bins=50, alpha=0.5, label='Training Data', edgecolor='black', color='blue')
plt.hist(test_preds_real, bins=50, alpha=0.5, label='Test Predictions', edgecolor='black', color='red')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.title('Price Distribution: Training vs Test Predictions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('histogram_comparison.png')
print("Graph 4 Saved: histogram_comparison.png")

# --- 7. SUBMISSION ---
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_preds_real
})
submission.to_csv('final_submission.csv', index=False)
print("Success! 'final_submission.csv' generated with optimized accuracy.")