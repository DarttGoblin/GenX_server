import os
import pandas as pd
from datasets import Dataset
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer
from diffusers import StableDiffusionPipeline
from diffusers import DDPMScheduler
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
import torch
from accelerate import Accelerator
from tqdm import tqdm

# Configuration
model_name = "CompVis/stable-diffusion-v1-4"
output_dir = "./genx_poster_model"
csv_path = "your_dataset.csv"  # Replace with your dataset path
image_column = "image"
text_columns = ["title", "description"]
image_size = 512
batch_size = 4
num_epochs = 10
learning_rate = 1e-5

# Load and preprocess dataset
df = pd.read_csv(csv_path)
df["prompt"] = df[text_columns].apply(lambda x: " ".join(x.astype(str)), axis=1)

# Define custom dataset
class PosterDataset(TorchDataset):
    def __init__(self, dataframe, tokenizer, image_transform):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row[image_column]).convert("RGB")
        image = self.image_transform(image)
        prompt = row["prompt"]
        tokenized = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        return {
            "pixel_values": image,
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
        }

# Initialize tokenizer and image transformations
tokenizer = CLIPTokenizer.from_pretrained(model_name)
image_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Create dataset and dataloader
dataset = PosterDataset(df, tokenizer, image_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model and scheduler
pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()

# Set up optimizer
optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=learning_rate)

# Initialize accelerator
accelerator = Accelerator()
pipeline.unet, optimizer, dataloader = accelerator.prepare(pipeline.unet, optimizer, dataloader)

# Training loop
pipeline.unet.train()
for epoch in range(num_epochs):
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        with accelerator.accumulate(pipeline.unet):
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)

            noise = torch.randn_like(pixel_values)
            timesteps = torch.randint(0, pipeline.scheduler.num_train_timesteps, (pixel_values.shape[0],), device=pixel_values.device).long()
            noisy_images = pipeline.scheduler.add_noise(pixel_values, noise, timesteps)

            noise_pred = pipeline.unet(noisy_images, timesteps, encoder_hidden_states=pipeline.text_encoder(input_ids)[0]).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

    # Save model checkpoint
    if accelerator.is_main_process:
        pipeline.save_pretrained(os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}"))
