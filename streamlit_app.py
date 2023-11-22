import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.transforms import Compose, ToTensor, ToPILImage

class GradCam:
    def __init__(self, model):
        self.model = model
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, x):
        image_size = (x.size(-1), x.size(-2))
        feature_maps = []
        for i in range(x.size(0)):
            img = x[i].data.cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)
            feature = x[i].unsqueeze(0)
            for name, module in self.model.named_children():
                if name == 'block8':
                    feature = module(feature)
                    feature.register_hook(self.save_gradient)
                    self.feature = feature
                elif name == 'last_linear':
                    feature = feature.view(feature.size(0), -1)
                    feature = module(feature)
                else:
                    feature = module(feature)
            classes = feature.sigmoid()
            one_hot, _ = classes.max(dim=-1)
            self.model.zero_grad()
            one_hot.backward()
            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
            mask = cv2.resize(mask.data.cpu().numpy(), image_size)
            mask = mask - np.min(mask)
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
            feature_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = feature_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            feature_maps.append(ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
        feature_maps = torch.stack(feature_maps)
        return feature_maps

# Classes
# classes = ['degawa', 'makken', 'wakasama', 'xxxx']
# classes = list(range(50))
classes = [
    'タモリ','明石家さんま','和田アキ子','ビートたけし','笑福亭鶴瓶','イチロー','黒柳徹子','デヴィ夫人','松本人志','マツコ・デラックス',
    '木村拓哉','小泉純一郎','浜田雅功','所ジョージ','中居正広','出川哲朗','羽生結弦','加藤茶','ダルビッシュ有','草彅剛',
    '福山雅治','長嶋一茂','桑田佳祐','フワちゃん','萩本欽一','浅田真央','内村光良','菅義偉','北島三郎','ヒロミ',
    '大坂なおみ','大谷翔平','劇団ひとり','蓮舫','泉ピン子','岸田文雄','香取慎吾','王貞治','長嶋茂雄','松田聖子',
    'IKKO','渡辺直美','研ナオコ','松平健','小池百合子','阿部寛','櫻井翔','クロちゃん','二宮和也','有吉弘行']

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(classify=True, num_classes=len(classes)).to(device).eval()
resnet.load_state_dict(torch.load('weight_1122.pth', map_location=device))
transform = Compose([
    np.float32,
    ToTensor()
])
mtcnn = MTCNN()
grad_cam = GradCam(resnet)

# Title
st.title('あなたに似ている有名人は？')

# Camera
buffer = st.camera_input('撮影された画像は保存されません')

if buffer is not None:
    img = Image.open(buffer)
    img_cropped = mtcnn(img)
    if img_cropped is None:
        st.error('顔が見つかりません！')
        st.stop()
    tensor_img = transform(img_cropped)
    tensor_img = torch.transpose(tensor_img,0,2)
    tensor_img = torch.transpose(tensor_img,0,1)
    tensor_img = tensor_img.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = resnet(tensor_img).softmax(dim=-1)
    pred_label = np.argsort(pred[0].cpu().numpy())[::-1]
    pred_prob = np.sort(pred[0].cpu().numpy())[::-1]
    df = pd.DataFrame(pred[0].cpu().numpy()*100, index=classes, columns=['score'])
    df.sort_values('score', ascending=False, inplace=True)

    feature_img = grad_cam(tensor_img).squeeze(0)
    feature_img = ToPILImage()(feature_img)

    fig_gradcam = plt.figure()
    ax = fig_gradcam.add_subplot(1, 1, 1)
    ax.imshow(feature_img)
    ax.axis('off')

    cols = st.columns(2)
    cols[0].pyplot(fig_gradcam, caption='Grad-CAM', use_column_width=True)
    for rank in range(3):
        cols[1].metric(label=f'似てる度第{rank+1}位', value=classes[pred_label[rank]], delta=f'{pred_prob[rank]*100:.1f}%')
    st.bar_chart(df)
    st.dataframe(df)