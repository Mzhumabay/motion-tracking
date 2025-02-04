import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Гиперпараметры
batch_size = 32
num_classes = 3  # Замените на количество ваших классов
learning_rate = 0.001
num_epochs = 10

# Трансформации данных
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Приведение к единому размеру
    transforms.ToTensor(),  # Преобразование в тензор
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Нормализация
])

# Загрузка данных
train_dataset = datasets.ImageFolder(r"C:\Users\mzhum\OneDrive\Pulpit\ai\project_track\dataset/train", transform=transform)
val_dataset = datasets.ImageFolder(r"C:\Users\mzhum\OneDrive\Pulpit\ai\project_track\dataset/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Проверка классов
class_names = train_dataset.classes
print(f"Классы: {class_names}")
# Загрузка предобученной модели
model = models.resnet18(pretrained=True)

# Замена последнего слоя на слой для вашей задачи
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Перенос на GPU, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    # Обучение
    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()

        # Прямой проход
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        # Обратный проход
        loss.backward()
        optimizer.step()

        # Точность
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()

    train_accuracy = train_correct / len(train_dataset)

    # Валидация
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()

    val_accuracy = val_correct / len(val_dataset)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss/len(train_loader):.4f}, "
          f"Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss/len(val_loader):.4f}, "
          f"Val Accuracy: {val_accuracy:.4f}")
torch.save(model.state_dict(), "model1.pth")
print("Модель сохранена!")
