import streamlit as st 
from PIL import Image
import numpy 
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
            
        self.fc1 = nn.Linear(64*15*15, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.drop = nn.Dropout2d(0.2)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1) 
        out = self.fc1(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.fc3(out)
        
        return out

@st.cache(allow_output_mutation=True)
def load_model():
  model=torch.load('model_1.pth')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Malaria Detection
         """
         )
train_transforms = transforms.Compose([transforms.Resize((120, 120)),
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                     ])
file = st.file_uploader("Please upload the image of Blood cell", type=["jpg", "png"])
def import_and_predict(image_data, model):
        model.eval()
        trans=train_transforms(image_data)
        test = trans.view(-1, 3, 120, 120)
        outputs = model(test)
        return outputs
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    predicted = torch.max(predictions, 1)[1]
    f=predicted.numpy()
    if (f[0]==0):
        st.text("The Blood cell is infected")
    else:
        st.text("The Blood cell is not infected")