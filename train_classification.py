import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
classes = ['cataract', 'diabetic_retinopathy', 'glaucoma','normal']
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. Dataset Class ---
class RetinalDataset(Dataset):
    def __init__(self, retinal_image_paths, mask_image_paths, labels, transform=None):
        """
        Args:
            retinal_image_paths (list): List of paths to retinal images.
            mask_image_paths (list): List of paths to corresponding mask images.
            labels (list): List of labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.retinal_image_paths = retinal_image_paths
        self.mask_image_paths = mask_image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.retinal_image_paths)

    def __getitem__(self, idx):
        retinal_image_path = self.retinal_image_paths[idx]
        mask_image_path = self.mask_image_paths[idx]
        label = self.labels[idx]

        retinal_image = Image.open(retinal_image_path).convert("RGB")
        mask_image = Image.open(mask_image_path).convert("L")  # Convert mask to grayscale


        if self.transform:
            retinal_image = self.transform(retinal_image)
            mask_image = self.transform(mask_image)  # Apply same transform to mask


        # Concatenate the retinal and mask images along the channel dimension
        # Corrected concatenation:  Make sure the mask image is a 3D tensor (C, H, W)
        # This is crucial because ToTensor() makes the RGB image (3, H, W).  The mask must match
        mask_image = mask_image.expand(3, -1, -1) # Expand grayscale to 3 channels

        combined_image = torch.cat((retinal_image, mask_image), dim=0)

        return combined_image, torch.tensor(label, dtype=torch.long) # ensure label is long for crossentropyloss


# --- 2. Data Loading and Preprocessing ---
def load_data(data_dir):
    """Loads image paths and labels from the directory structure."""
    retinal_image_paths = []
    mask_image_paths = []
    labels = []

    # classes = ['cataract', 'normal', 'diabetic_retinopathy', 'glaucoma']
    class_to_idx = {c: i for i, c in enumerate(classes)} # creates a mapping like {'cataract': 0, 'normal': 1,...}


    for disease in classes:
        retinal_dir = os.path.join(data_dir, disease, 'retinal_images')
        mask_dir = os.path.join(data_dir, disease, 'masked_images')

        for filename in os.listdir(retinal_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                retinal_image_paths.append(os.path.join(retinal_dir, filename))
                mask_image_paths.append(os.path.join(mask_dir, filename)) # Assume same filename
                labels.append(class_to_idx[disease])  # Use the mapping

    return retinal_image_paths, mask_image_paths, labels, classes

# --- 3. Model Definition ---
class CombinedImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CombinedImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)  # Input channels = 6 (RGB + RGB mask)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive average pooling to handle varying feature map sizes
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8)) #adjust these numbers as needed

        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Adjust the input size accordingly
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)



    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the feature map
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x



# --- 4. Training Loop ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path='saved_models'):
    """Trains the model."""
    best_val_acc = 0.0
    os.makedirs(save_path, exist_ok=True)  # Create save directory if it doesn't exist


    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]")
        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0) # Corrected: multiply by batch size
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / total_samples

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')


        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0

        loop = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} [VALID]")
        with torch.no_grad():
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0) # Corrected: multiply by batch size
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct_predictions / val_total_samples

        print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

        # Save the model if validation accuracy improves
        if val_epoch_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_epoch_acc:.4f}. Saving model...")
            best_val_acc = val_epoch_acc
            model_path = os.path.join(save_path, f'model_epoch_{epoch+1}_val_acc_{val_epoch_acc:.4f}.pth')
            torch.save(model.state_dict(), model_path)  # Save only the model's state dictionary
            print(f"Model saved to {model_path}")

    print('Finished Training')
    print(f"Best validation accuracy: {best_val_acc:.4f}")


# --- 5. Main Script ---
if __name__ == '__main__':
    # Set the path to your data directory
    data_dir = '/content/CombinedData'  # Replace with the actual path

    # Load data
    retinal_image_paths, mask_image_paths, labels, classes = load_data(data_dir)
    num_classes = len(classes)

    # Split data into training and validation sets
    retinal_train_paths, retinal_val_paths, mask_train_paths, mask_val_paths, train_labels, val_labels = train_test_split(
        retinal_image_paths, mask_image_paths, labels, test_size=0.2, random_state=42, stratify=labels) #stratify ensures equal representation of classes

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5]) # Normalizing on all 4 channels requires knowing stats on mask channel too, so commented out.  Consider it.
    ]) #added transforms

    # Create datasets
    train_dataset = RetinalDataset(retinal_train_paths, mask_train_paths, train_labels, transform=transform)
    val_dataset = RetinalDataset(retinal_val_paths, mask_val_paths, val_labels, transform=transform)

    # Create data loaders
    batch_size = 8 # Reduced batch size to avoid memory issues on GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Instantiate the model
    model = CombinedImageClassifier(num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 30
    save_path = '/content/drive/MyDrive/saved_models/my_model'  # Directory to save models
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path)