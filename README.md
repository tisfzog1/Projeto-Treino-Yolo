# YOLOv8 com COCO128 no Google Colab

Este repositório contém um notebook do Google Colab para treinar o modelo **YOLOv8** usando o dataset **COCO128**.  
O fluxo inclui preparação do dataset, separação automática em treino/validação, criação do `dataset.yaml`, treinamento do modelo e inferência em imagens de teste.

---

## 🚀 Executar no Google Colab

Clique no botão abaixo para abrir o notebook diretamente no Google Colab (GPU recomendada):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/tisfzog1/Projeto-Treino-Yolo/blob/main/TreinoYolo.ipynb)

---

## 📖 Passo a Passo

### **Etapa 1 — Montar Google Drive**
O notebook monta o Google Drive e define a pasta do projeto:
```python
from google.colab import drive
import os

drive.mount('/content/drive')
project_dir = "/content/drive/MyDrive/yolo_project"
os.makedirs(project_dir, exist_ok=True)
%cd $project_dir
```

---

### **Etapa 2 — Instalar dependências**
Instale o YOLOv8 da Ultralytics e atualize o `pip`:
```bash
!pip install -U pip
!pip install ultralytics
```

---

### **Etapa 3 — Baixar e preparar o dataset**
Baixa automaticamente o **COCO128** e separa **20% das imagens para validação** (o dataset oficial só vem com `train2017`):
```python
import zipfile, shutil, random, os

# Baixar e extrair
!curl -L https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip -o coco128.zip
with zipfile.ZipFile('coco128.zip', 'r') as zip_ref:
    zip_ref.extractall(project_dir)

# Separar treino/validação
coco_path = "/content/drive/MyDrive/yolo_project/coco128"
train_dir = os.path.join(coco_path, "images/train2017")
val_dir = os.path.join(coco_path, "images/val")
os.makedirs(val_dir, exist_ok=True)

images = [f for f in os.listdir(train_dir) if f.endswith(".jpg")]
random.shuffle(images)
val_images = images[:int(0.2 * len(images))]

for img in val_images:
    shutil.move(os.path.join(train_dir, img), os.path.join(val_dir, img))

print("✅ Dataset pronto!")
```

---

### **Etapa 4 — Criar o `dataset.yaml`**
O notebook gera automaticamente o arquivo de configuração:
```python
yaml_path = os.path.join(coco_path, 'dataset.yaml')

with open(yaml_path, 'w') as f:
    f.write("""train: /content/drive/MyDrive/yolo_project/coco128/images/train2017
val: /content/drive/MyDrive/yolo_project/coco128/images/val

nc: 80
names: ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
        'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
        'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
        'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
        'baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
        'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
        'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
        'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
        'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors',
        'teddy bear','hair drier','toothbrush']""")
```

---

### **Etapa 5 — Treinar YOLOv8**
Agora é só rodar o treinamento:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # modelo pequeno e rápido

results = model.train(
    data=yaml_path,
    epochs=10,
    imgsz=640,
    batch=16,
    project=project_dir,
    name='coco128'
)
```

---

### **Etapa 6 — Fazer inferência em imagens de teste**
Faça upload de imagens no Colab e rode a predição:
```python
from google.colab import files
uploaded_test = files.upload()

test_dir = os.path.join(project_dir, "test_images")
os.makedirs(test_dir, exist_ok=True)

for filename in uploaded_test.keys():
    with open(os.path.join(test_dir, filename), 'wb') as f:
        f.write(uploaded_test[filename])

# Rodar inferência
model_path = os.path.join(project_dir, "coco128", "weights", "best.pt")
!yolo detect predict model={model_path} source={test_dir} save=True
```

As imagens com predições serão salvas automaticamente no Google Drive.

---

## ⚙️ Requisitos
- Conta no [Google Colab](https://colab.research.google.com/)  
- GPU ativada no Colab:  
  *Menu → Executar → Alterar tipo de ambiente de execução → GPU*

---

## 📜 Licença
Este projeto segue a licença **GPL-3.0**, conforme o dataset COCO128.

---

## ✨ Créditos
- [Ultralytics](https://github.com/ultralytics) pelo YOLOv8 e dataset COCO128.
