import torch
from app.sana_sprint_pipeline import SanaSprintPipeline
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

sana = SanaSprintPipeline("configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml")
sana.from_pretrained("hf://Efficient-Large-Model/Sana_Sprint_1.6B_1024px/checkpoints/Sana_Sprint_1.6B_1024px.pth")

prompt = "a tiny astronaut hatching from an egg on the moon"

image = sana(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
    num_inference_steps=2,
    generator=generator,
)
save_image(image, 'sana_sprint.png', nrow=1, normalize=True, value_range=(-1, 1))
