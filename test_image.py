import cv2
import torch
import torch.nn
import argparse
from myresnet import MyResNet
from torchvision.transforms import Compose,ToPILImage,Resize,ToTensor

def get_args():
    parser = argparse.ArgumentParser(description="Test image model")
    parser.add_argument("--image_path","-i",type = str, help = "path to image", required= True)
    parser.add_argument("--model_detection_path", "-d", type=str,default="best.pt",help="Path to model detection")
    parser.add_argument("--model_cls_path", "-cls", type=str, default="trained_models/best_cnn.pt",help = "Load from this checkpoint")
    parser.add_argument("--conf_threshold", "-c", type=float, default=0.5,help = "Confident threshold")
    args = parser.parse_args()
    return args

# image_path = "test4.jpg"
# model_detection_path = "best.pt"
# model_cls_path = "trained_models/best_cnn.pt"
# conf_threshold = 0.5

class_map = {
    '0': {'name': 'Female', 'color': (0,235,42)},
    '1': {'name': 'Male', 'color': (206,255,0)}
}

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_detection = torch.hub.load('ultralytics/yolov5','custom',path = args.model_detection_path,source='github')
    model_detection.conf = min(0.25, args.conf_threshold)
    model_detection.iou = 0.45
    model_cls = MyResNet(num_classes=2)
    checkpoint = torch.load(args.model_cls_path,map_location=device)
    model_cls.load_state_dict(checkpoint["model"])
    model_cls.to(device)
    model_cls.eval()

    transform = Compose([
        ToPILImage(),
        Resize((224,224)),
        ToTensor()
    ])

    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image,cv2.COLOR_BGR2RGB)
    results = model_detection(image)
    detections = results.xyxy[0].cpu().numpy()
    h_img, w_img, _ = image.shape

    for detection in detections:
        print(detection)
        x1,y1,x2,y2,confidence,cls_id = detection
        # x1,y1,x2,y2,cls_id = int(x1),int(y1),int(x2),int(y2),int(cls_id)
        x1 = max(0,int(x1))
        y1 = max(0,int(y1))
        x2 = min(w_img,int(x2))
        y2 = min(h_img,int(y2))
        cls_id = int(cls_id)
        if confidence < args.conf_threshold: continue
        if x2-x1 < 5 or y2-y1 < 5: continue
        face_crop = image[y1:y2,x1:x2]
        input_tensor = transform(face_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model_cls(input_tensor)
            probs = torch.softmax(outputs,dim=1)
            conf, pred_cls = torch.max(probs,1)

            cls_idx_str = str(pred_cls.item())
            cls_info = class_map.get(cls_idx_str,{'name':'Unknown', 'color': (255,255,255)})

            name = cls_info['name']
            color = cls_info['color']

            label = f"{name} {conf.item():.2f}"
            cv2.rectangle(ori_image,(x1,y1), (x2,y2),color,2)
            cv2.putText(ori_image,label,(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX,0.9,color,2)

    cv2.imshow("Output",ori_image)
    cv2.waitKey(0)    
    cv2.imwrite("output.jpg",ori_image)

if __name__ == "__main__":
    args = get_args()
    main(args)
