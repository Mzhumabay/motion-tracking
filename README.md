# Отслеживание движения и классификация

Этот проект демонстрирует систему отслеживания движения и классификации объектов с использованием предварительно обученной нейронной сети и методов компьютерного зрения. Система способна обнаруживать движущиеся объекты на статичном видео, классифицировать их типы и улучшать их видимость за счёт изменения яркости и контраста. В качестве классов для модели были выбраны персонажи из игры "Valorant". Сбор датасета для модели было реализовано полуавтоматический.

## Основные возможности

Обнаружение движения:

Использует разницу кадров для определения движущихся объектов на статичном видео.

Удаление шума и улучшение маски с помощью морфологических операций.

Классификация объектов:

Применяет предварительно обученную модель ResNet-18, дообученную для определённых классов объектов.

Классифицирует обнаруженные объекты на предопределённые категории с вероятностями.

Обработка ограничивающих рамок (Bounding Boxes):

Извлекает ограничивающие рамки для обнаруженных объектов.

Отзеркаливает содержимое рамок при необходимости.

Регулировка яркости и контраста:

Улучшает видимость объектов в пределах ограничивающих рамок программно.

## В работе были Использованы библиотеки: PyTorch, OpenCV, ResNet-18 Model

## Обозначение файлов:
track.py - программа для  трекинга объектов на видео

model.py - программа для обучения модели классификаций объектов

model.pth - модель классификаций объектов, результат model.py

track_video.avi - результат трекинга в виде видео где объекты выделены прямоугольниками

dataset_from_video - программа для сбора датасета из видео

video_val - папка из видео с объектами 

