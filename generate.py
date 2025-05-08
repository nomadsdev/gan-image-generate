import torch
import os
from gan_model import Generator
from torchvision.utils import save_image

def generate_images(num_images=16, latent_dim=100, channels=3, model_path="saved_models/generator.pth"):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize generator
    generator = Generator(latent_dim, channels).to(device)
    
    # Load trained model
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained model")
    else:
        print("No trained model found!")
        return
    
    # Generate images
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        gen_imgs = generator(z)
        
        # Save generated images
        os.makedirs("generated_images", exist_ok=True)
        save_image(gen_imgs.data, "generated_images/generated.png", nrow=4, normalize=True)
        print(f"Generated {num_images} images and saved to generated_images/generated.png")

if __name__ == "__main__":
    generate_images() 