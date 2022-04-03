import clip
import torch

def _process_text_prompts(prompts, model_clip, device):
    output = []

    if not (prompts is None):
        for prompt, weight in prompts:
            t = clip.tokenize([prompt]).to(device)
            output.append((model_clip.encode_text(t).detach(), torch.tensor(weight, device=device)))

    return output

def _process_image_prompts(prompts, model_clip, clip_preprocessor, device):
    output = []

    if not (prompts is None):
        for img, weight in prompts:
            img = clip_preprocessor(img).unsqueeze(0).to(device)
            output.append((model_clip.encode_image(img), torch.tensor(weight, device=device)))

    return output

class Objective:
    def __init__(self, text_prompts, model_clip, device, agumenter, clip_preprocessor,
                 image_prompts = None,
                 exclude_text_prompts = None, exclude_image_prompts = None):

        self.text_embeddings = _process_text_prompts(text_prompts, model_clip, device)
        self.image_embeddings = _process_image_prompts(image_prompts, model_clip, clip_preprocessor, device)
        self.exclude_text_embeddings = _process_text_prompts(exclude_text_prompts, model_clip, device)
        self.exclude_image_embeddings = _process_image_prompts(exclude_image_prompts, model_clip, clip_preprocessor, device)
        self.device = device
        self.model = model_clip
        self.agumenter = agumenter

    def __call__(self, img):
        img = self.agumenter(img)
        img_encoding = self.model.encode_image(img)

        loss = torch.tensor(0.0, device=self.device)

        for embedding, weight in self.text_embeddings:
            loss += torch.cosine_similarity(img_encoding, embedding, -1).mean() * -1 * weight

        for embedding, weight in self.image_embeddings:
            loss += torch.cosine_similarity(img_encoding, embedding, -1).mean() * -1 * weight

        for embedding, weight in self.exclude_text_embeddings:
            loss += torch.cosine_similarity(img_encoding, embedding, -1).mean() * weight

        for embedding, weight in self.exclude_image_embeddings:
            loss += torch.cosine_similarity(img_encoding, embedding, -1).mean() * weight

        return loss
