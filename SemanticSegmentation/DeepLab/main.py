import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


# See if GPU is available and if yes, use it
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the standard transforms that need to be done at inference time
imagenet_stats = [[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]]
preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(mean = imagenet_stats[0],
                                                                            std  = imagenet_stats[1])])


def get_pred(img, model, width, height, channels, background):
    # Resize the image to a better performing size
    # img = cv2.resize(img, (256, 256))

    input_tensor = preprocess(img).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # Make the predictions for labels across the image
    with torch.no_grad():
      output = model(input_tensor)["out"][0]
      output = output.argmax(0)

    labels = output.cpu().numpy()

    # Wherever there's empty space/no person, the label is zero 
    # Hence identify such areas and create a mask (replicate it across RGB channels)
    mask = labels == 0
    mask = np.repeat(mask[:, :, np.newaxis], channels, axis = 2)

    # Remove the background
    # img[mask] = 0

    # Change the background to the image of choice
    background = cv2.resize(background, (height, width))
    img[mask] = background[mask]

    # Define the kernel size for applying Gaussian Blur
    # blur_value = (51, 51)
    # Apply the Gaussian blur for background with the kernel size specified in constants above
    # blur = cv2.GaussianBlur(img, blur_value, 0)
    # img[mask] = blur[mask]

    # Resize the image back to original size
    # img = cv2.resize(img, (height, width))

    return img


def load_model():
    # Load the DeepLab v3 model to system
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    model.to(device).eval()
    return model


def show_webcam(model, mirror=False):
    # Background image
    background = cv2.imread("./images/background.png")

    cam = cv2.VideoCapture(0)
    ret_val, img = cam.read()
    width, height, channels = img.shape
    while ret_val:

        img = cv2.flip(img, 1)
        img = get_pred(img, model, width, height, channels, background)
        cv2.imshow('my webcam', img)
        ret_val, img = cam.read()
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    model = load_model()
    show_webcam(model)


if __name__ == "__main__":
    main()
