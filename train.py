import os.path
import torch
import cv2
from face_cls_dataset import FaceDataset
from myresnet import MyResNet,BasicBlock
from torchvision.transforms import Compose , ToTensor , Resize, RandomAffine,ColorJitter
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn 
from torch.optim import SGD
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from argparse import ArgumentParser
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import shutil

def get_args():
    parser = ArgumentParser(description="training")
    parser.add_argument("--root","-r", type= str, default="./dataset", help="Root")
    parser.add_argument("--epochs","-e", type= int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size","-b", type= int, default=8, help="Number of epochs")
    parser.add_argument("--image-size","-i", type= int, default=224, help="Size of image")
    parser.add_argument("--logging","-l", type= str, default="tensorboard")
    parser.add_argument("--trained_models","-t", type= str, default="trained_models")
    parser.add_argument("--checkpoint","-c", type= str, default=None)
    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_transform = Compose([
        RandomAffine(
            degrees=(-5,5),
            translate=(0.05,0.15),
            scale =(0.85,1.15),
            shear=10
        ),
        Resize((args.image_size,args.image_size)),
        ColorJitter(
            brightness=0.5  ,
            contrast=0.5,
            saturation=0.25,
            hue= 0.05   
        ),
        ToTensor()
    ])
    test_transform = Compose([
        Resize((args.image_size,args.image_size)),
        ToTensor()
    ])
    # 2. Tạo một dataset GỐC với train_transform
    # Chúng ta sẽ dùng nó để lấy train_dataset VÀ các chỉ số (indices) cho test
    data_with_train_transform = FaceDataset(root=args.root, transform=train_transform)

    # 3. Tính toán kích thước
    data_size = len(data_with_train_transform)
    train_size = int(0.9 * data_size)
    test_size = data_size - train_size

    # 4. Thực hiện split
    # 'train_dataset' được tạo ra từ đây là ĐÚNG
    # 'test_subset_wrong_transform' là một Subset chứa các CHỈ SỐ test,
    # nhưng nó đang trỏ đến data có train_transform (SAI)
    train_dataset, test_subset_wrong_transform = random_split(
        data_with_train_transform, 
        [train_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )

    # 5. Tạo một dataset MỚI chỉ dành cho test, với test_transform
    data_with_test_transform = FaceDataset(root=args.root, transform=test_transform)

    # 6. Tạo test_dataset cuối cùng (ĐÚNG)
    # Bằng cách kết hợp data có transform đúng (data_with_test_transform)
    # với các chỉ số test (lấy từ test_subset_wrong_transform.indices)
    test_dataset = Subset(data_with_test_transform, test_subset_wrong_transform.indices)
    train_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size= args.batch_size,
        num_workers= 4,
        shuffle= True,
        drop_last= True
    ) 
    test_dataloader = DataLoader(
        dataset= test_dataset,
        batch_size= args.batch_size,
        num_workers= 4,
        drop_last= True,
        shuffle= False
    )
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)
    writer = SummaryWriter(args.logging)
    model = MyResNet(num_classes=2)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr = 1e-3, momentum= 0.9)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_accuracy = 0
        
    num_iters = len(train_dataloader) 
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour= "cyan")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch+1, args.epochs, iter+1, num_iters, loss_value))
            writer.add_scalar("Train/Loss", loss_value, epoch*num_iters +iter)
            
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions.cpu(), dim = 1)
                all_predictions.extend(indices)
                loss_value = criterion(predictions, labels)
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), class_names=data_with_train_transform.categories, epoch= epoch)
        accuracy = accuracy_score(all_labels, all_predictions)
        print("Epoch {}: Accuracy: {}".format(epoch + 1, accuracy_score(all_labels, all_predictions)))
        writer.add_scalar("Val/Accuracy", accuracy, epoch)
        # torch.save(model.state_dict(), "{}/last_cnn.pt".format(args.trained_models))
        checkpoint = {
            "epoch": epoch + 1,
            "best_acc": best_accuracy,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpoint = {
            "epoch": epoch + 1,
            "best_acc": best_accuracy,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))