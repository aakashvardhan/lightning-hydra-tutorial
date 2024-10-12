import os
import argparse
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms


import lightning as L
from PIL import Image
import matplotlib.pyplot as plt
from datamodules.dog_breed_datamodule import DogBreedDataModule
from models.dog_breed_classifier import DogBreedClassifier

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


CLASS_LABELS = [
    "Beagle",
    "Boxer",
    "Bulldog",
    "Dachshund",
    "German_Shepherd",
    "Golden_Retriever",
    "Labrador_Retriever",
    "Poodle",
    "Rottweiler",
    "Yorkshire_Terrier",
]


def img_denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor image or batch of images with mean and standard deviation.

    Args:
        tensor (torch.Tensor): Input tensor image(s) (3D or 4D).

    Returns:
        torch.Tensor: Denormalized tensor image(s) with values between 0 and 1.
    """
    # Constants for normalization
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])

    # Ensure tensor is on CPU
    tensor = tensor.cpu()

    # Handle both single images and batches
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    # Denormalize
    denormalized = tensor * STD[None, :, None, None] + MEAN[None, :, None, None]
    return torch.clamp(denormalized, 0, 1)


def inference(model, img_tensor):
    """
    Perform inference on an image tensor.

    Args:
        model (torch.nn.Module): The model to use for inference.
        img_tensor (torch.Tensor): The image tensor to perform inference on.

    Returns:
        Predicted class label and confidence score.
    """
    # Set model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    predicted_label = CLASS_LABELS[predicted_class]
    confidence = probabilities[0][predicted_class].item()

    dict_score = {"predicted_label": predicted_label, "confidence": confidence}

    return dict_score


def save_predictions(
    imgs,
    actual_labels,
    predicted_labels,
    confidence_scores,
    save_dir: str,
):
    """
    Save plt of 10 images with predicted and actual labels.
    """
    num_images = imgs.shape[0]
    denormalized_imgs = img_denormalize(imgs)

    plt.figure(figsize=(20, 4 * ((num_images + 4) // 5)))
    for i in range(num_images):
        plt.subplot(((num_images + 4) // 5), 5, i + 1)
        img = denormalized_imgs[i].permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        plt.title(
            f"Actual: {actual_labels[i]}\nPredicted: {predicted_labels[i]}\nConfidence: {confidence_scores[i]:.4f}"
        )
        plt.axis("off")

    plt.savefig(save_dir)
    plt.close()


def main(args):
    """
    Main function to perform inference on 10 random test images.
    """
    # Load model
    model = DogBreedClassifier.load_from_checkpoint(args.checkpoint_path)
    model.to(device)

    # Create a directory to save the predictions
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    datamodule = DogBreedDataModule(args.data_dir, args.batch_size, args.num_workers)
    datamodule.setup(stage="test")

    # Get the test dataset
    test_dataset = datamodule.test_dataset

    # Select 10 random images from the test dataset
    num_samples = min(10, len(test_dataset))
    random_indices = random.sample(range(len(test_dataset)), num_samples)

    selected_imgs = []
    actual_labels = []
    predicted_labels = []
    confidence_scores = []

    model.eval()
    with torch.no_grad():
        for idx in random_indices:
            img, label = test_dataset[idx]
            img = img.unsqueeze(0).to(device)

            result = inference(model, img)

            selected_imgs.append(img.squeeze(0))
            actual_labels.append(CLASS_LABELS[label])
            predicted_labels.append(result["predicted_label"])
            confidence_scores.append(result["confidence"])

    # Stack the selected images
    selected_imgs = torch.stack(selected_imgs)

    # Save predictions
    save_predictions(
        selected_imgs,
        actual_labels,
        predicted_labels,
        confidence_scores,
        os.path.join(args.save_dir, "test_predictions.png"),
    )

    print(f"Test predictions saved to {args.save_dir}")

    # Print classification results
    print("\nClassification Results:")
    for actual, predicted, confidence in zip(
        actual_labels, predicted_labels, confidence_scores
    ):
        print(f"Actual: {actual}, Predicted: {predicted}, Confidence: {confidence:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference on dog breed images."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/", help="Directory containing the dataset"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="predictions/",
        help="Directory to save predictions",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )

    args = parser.parse_args()
    main(args)
