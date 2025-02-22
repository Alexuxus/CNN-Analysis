from flask import Flask, request, render_template, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import clip

app = Flask(__name__)

# Доступные модели классификации (ResNet, VGG и т.д.)
MODEL_NAMES = {
    "resnet18": models.resnet18,
    "vgg16": models.vgg16,
    # Добавьте другие модели по мере необходимости
}

# Инициализация модели классификации по умолчанию
current_model_name = "resnet18"
model = models.resnet18(pretrained=True)
model.eval()

# Инициализация CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")  # Или "cuda", если есть GPU
clip_model.eval()

# Преобразование изображения для ResNet
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Определение класса изображения с помощью ResNet
def predict(image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Получение описания изображения с помощью CLIP
def get_clip_description(image, possible_classes):
    image = clip_preprocess(image).unsqueeze(0).to("cpu")  # Или "cuda"
    text = clip.tokenize(possible_classes).to("cpu") # Или "cuda"

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze()

        values, indices = torch.topk(similarity, 5) #Топ 5 классов

    return [possible_classes[index] for index in indices] #Возвращает названия классов


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global model, current_model_name

    if request.method == 'POST':
        file = request.files['file']
        model_name = request.form.get('model_select')

        if model_name and model_name in MODEL_NAMES and model_name != current_model_name:
            try:
                model = MODEL_NAMES[model_name](pretrained=True)
                model.eval()
                current_model_name = model_name
                print(f"Model changed to {current_model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                return jsonify({'error': f"Ошибка при загрузке модели {model_name}: {e}"}), 500

        if file:
            try:
                image = Image.open(file.stream).convert('RGB')
                image_tensor = transform_image(image)

                # Задаём возможные классы для CLIP (их можно сделать настраиваемыми)
                possible_classes = ["cat", "dog", "bird", "car", "tree", "house", "person", "sky", "grass", "water"]

                # Получаем описание от CLIP
                clip_descriptions = get_clip_description(image, possible_classes)

                # Получаем предсказание от ResNet (только индекс класса)
                resnet_prediction_index = predict(image_tensor)

                return jsonify({
                    'resnet_index': resnet_prediction_index,
                    'clip_descriptions': clip_descriptions
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 400

        return jsonify({'error': 'No file uploaded'}), 400

    return render_template('index.html', models=MODEL_NAMES.keys(), current_model=current_model_name)

if __name__ == '__main__':
    app.run(debug=True)