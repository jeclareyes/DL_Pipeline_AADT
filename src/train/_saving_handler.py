import torch
from omegaconf import OmegaConf

def save_checkpoint(model, optimizer, epoch, loss, cfg, filename="checkpoint.pt"):
    """Guarda el estado del entrenamiento y la configuración."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        # Guardamos la config dentro del checkpoint para trazabilidad total
        'config': OmegaConf.to_container(cfg, resolve=True)
    }
    torch.save(checkpoint, filename)