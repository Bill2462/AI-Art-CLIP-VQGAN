import os
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from omegaconf import OmegaConf
from taming.models.cond_transformer import Net2NetTransformer
from taming.models.vqgan import VQModel
import clip
from IPython import display
from PIL import ImageFile, Image

def get_device(use_gpu):
    if use_gpu:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    return torch.device(device)

def load_vqgan(checkpoint_filepath, config_filepath, device,
               model_type="conditional"):

    conf = OmegaConf.load(config_filepath)

    if model_type == "conditional":
        model = Net2NetTransformer(**conf.model.params)
        model.init_from_ckpt(checkpoint_filepath)
        model.eval().requires_grad_(False)
        model = model.first_stage_model.to(device)

    elif model_type == "uncoditional_vqgan":
        model = VQModel(**conf.model.params)
        model.init_from_ckpt(checkpoint_filepath)
        model.eval().requires_grad_(False).to(device)

    else:
        raise ValueError("Invalid model type, only 'conditional' and 'uncoditional_vqgan' are supported")

    return model

def load_clip(model_filepath, device):
    model, preprocessor = clip.load(model_filepath, device=device, jit=False)
    model = model.eval().requires_grad_(False)

    return model, preprocessor

def initialize_z(width, height, model, device):
    f = 2**(model.decoder.num_resolutions - 1)
    n_toks = model.quantize.n_e
    e_dim = model.quantize.e_dim

    input_x = width // f
    input_y = height // f

    indexes = torch.randint(n_toks, [input_x * input_y], device=device)
    one_hot = F.one_hot(indexes, n_toks).float()
    z = one_hot @ model.quantize.embedding.weight

    return z.view([-1, input_y, input_x, e_dim]).permute(0, 3, 1, 2)

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply

def vector_quantize(z, codebook):
    d = z.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * z @ codebook.T
    indices = d.argmin(-1)
    z_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(z_q, z)

def generate(z, model):
    z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return model.decode(z_q).add(1).div(2).clamp(0, 1)

def run(opt, z, objective, model_vqgan, model_clip, epochs, save_interval=100, save_path="img"):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(epochs):
        # Clear previous gradient values.
        opt.zero_grad()

        # Generate image using VQGAN.
        img = generate(z, model_vqgan)

        # Compute loss using objective function.
        loss = objective(img)

        # Compute derivative of input to VQGAN with respect to loss.
        loss.backward()

        # Gradient clipping so we do not have gradient explosion (pink output image).
        torch.nn.utils.clip_grad_norm_(z, 0.1)

        # Step optimizer.
        opt.step()

        # Save image every 100 epochs and update preview.
        if i % save_interval == 0 or i == epochs-1:
            display.clear_output(wait=True)

            loss = float(loss.detach().cpu())
            print(f"epoch: {i}, loss: {loss}")

            # Convert image to PIL object.
            img_pil = TF.to_pil_image(img[0])

            image_filepath = os.path.join(save_path, f"{i}.png")

            # Save it to drive.
            img_pil.save(image_filepath)

            # Display it
            display.display(display.Image(image_filepath))
