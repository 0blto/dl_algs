import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from dllib.trainers.vae_trainer import VAETrainer

# Сначала выбираем backend (CPU/GPU), потом тянем dllib, чтобы все слои
# сразу получили правильный xp.
try:
    import cupy as cp
    print("CuPy успешно загружен. Используется GPU ускорение.")
    USE_GPU = True
    print(f"GPU устройство: {cp.cuda.Device().compute_capability}")
    print(f"Память GPU: {cp.cuda.Device().mem_info[1] / 1024**3:.2f} GB")
except ImportError:
    print("CuPy не установлен. Используется CPU (NumPy).")
    import numpy as cp
    USE_GPU = False

from dllib.utils import device

device.configure_backend(cp, USE_GPU)
device.seed_everything(884736743)
xp = device.xp

# =======================
# Импорты твоих классов
# =======================
from dllib.model.vae import VAEEncoder, VAEDecoder
from dllib.model.gan import Generator, Discriminator
from dllib.trainers.vae_gan_trainer import VAEGANTrainer

# =======================
# Загружаем локальный MNIST
# =======================
mnist_path = Path(__file__).parent / "mnist.npz"
data = np.load(mnist_path)
x_train = data["x_train"].astype(np.float32) / 255.0 * 2 - 1.0
x_test = data["x_test"].astype(np.float32) / 255.0 * 2 - 1.0

# Меняем форму к (N, 1, 28, 28)
x_train = x_train[:, np.newaxis, 2:26, 2:26]
x_test = x_test[:, np.newaxis, 2:26, 2:26]
#
# Нормализация уже сделана выше (деление на 255), повторно не делим,
# только приводим к валидному диапазону.
x_train = np.clip(x_train.astype(np.float32), -1.0, 1.0)
x_test = np.clip(x_test.astype(np.float32), -1.0, 1.0)

print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")

# =======================
# Создаём модели
# =======================
latent_dim = 64
encoder = VAEEncoder(input_depth=1, latent_dim=latent_dim)
decoder = VAEDecoder(output_depth=1, latent_dim=latent_dim)
discriminator = Discriminator(input_depth=1)

# =======================
# Создаём тренера
# =======================
trainer = VAEGANTrainer(encoder, decoder, discriminator)

# =======================
# Тренировка на всём датасете
# =======================
history = trainer.train(
    x_train=x_train,
    epochs=30,
    batch_size=128,
)

# =======================
# Проверяем реконструкцию
# =======================
sample_batch = x_test[:16]
z = encoder.forward(sample_batch)
reconstructions = decoder.forward(z)

for i in range(min(4, len(sample_batch))):
    plt.subplot(2, 4, i + 1)
    plt.imshow(sample_batch[i, 0], cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 4, i + 5)
    plt.imshow(reconstructions[i, 0], cmap="gray")
    plt.title("Reconstruction")
    plt.axis("off")

plt.show()
