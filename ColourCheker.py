import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# for label the data -> pip install label-studio package
# after run label-studio
# username and password is matteo.griot.work@gmail.com and admin_gri_97
# https://www.youtube.com/watch?v=A1V8yYlGEkI&t=161s
SCORE_THRESHOLD = 0.8

def get_model_paths(model_type):
    model_path = os.path.join(os.getcwd(), "models", model_type)
    model_file = os.path.join(model_path, f"{model_type}.pt")
    return model_path, model_file

def training_(model_type, overwrite=False):
    model_path, model_file = get_model_paths(model_type)
    
    # Create the directory if it does not exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Check if the model already exists
    if not os.path.exists(model_file) or overwrite:
        # Create a new YOLO model from scratch
        model = YOLO("yolo11n.yaml")

        # Load a pretrained YOLO model (recommended for training)
        model = YOLO("yolo11n.pt")
        model.train(
            data=os.path.join(model_path, "dataset.yaml"),
            batch=8,
            epochs=1000,
            imgsz=640,
            workers=0,
            project=model_path,
            name="train_results",
        )

        # Validate the model
        model.val(project=model_path, name="val_results")

        # Save the model
        model.save(model_file)
    else:
        print("Model already exists. Skipping training.")

def load_model(model_path):
    return YOLO(model_path)

def predict(model, image_path):
    image = Image.open(image_path)
    image_np = np.array(image)
    if image_np.ndim == 2:  # If the image is grayscale, convert it to RGB
        image_np = np.stack((image_np,) * 3, axis=-1)
    elif image_np.shape[2] == 4:  # If the image has an alpha channel, remove it
        image_np = image_np[:, :, :3]
    results = model.predict(image_np, imgsz=640)
    return results

def plot_results(image_path, results):
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for result in results[0].boxes:  # Access the first batch of results
        box = result.xyxy[0].cpu().numpy()  # Get the bounding box coordinates
        score = result.conf[0].cpu().numpy()  # Get the confidence score
        label = int(result.cls[0].cpu().numpy())  # Get the class label
        color = "green" if score > SCORE_THRESHOLD else "red"
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        plt.text(
            box[0],
            box[1] - 10,
            f"{label}: {score:.2f}",
            color=color,
            fontsize=12,
            weight="bold",
        )

    plt.show()

def crop_object(image_path, results):
    image = Image.open(image_path)
    for result in results[0].boxes:  # Access the first batch of results
        box = result.xyxy[0].cpu().numpy()  # Get the bounding box coordinates
        cropped_image = image.crop((box[0], box[1], box[2], box[3]))
        return cropped_image

def plot_cropped_image(cropped_image):
    plt.imshow(cropped_image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    model_type = input("Enter the model type (e.g., ColourChecker, ColoredSquares): ").strip()
    overwrite = input("Do you want to overwrite the existing model? (yes/no): ").strip().lower() == "yes"
    training_(model_type, overwrite=overwrite)
    model_path, model_file = get_model_paths(model_type)
    model = load_model(model_file)
    sample_image_path = input("Enter the path to the sample image: ")
    results = predict(model, sample_image_path)
    print(results)
    plot_results(sample_image_path, results)
    cropped_image = crop_object(sample_image_path, results)
    plot_cropped_image(cropped_image)
