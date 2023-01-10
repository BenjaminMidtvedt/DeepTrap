# %%

import numpy as np

IMAGE_SIZE = 128

#%%

def gerchberg_saxton_phase_retrieval(source, target, error_threshold=1e-3, max_iterations=1000):
    """Gerchberg-Saxton phase retrieval algorithm.

    Parameters
    ----------
    source : array_like
        The source pattern (the amplitude distribution of the input beam).
    target : array_like
        The target pattern (the desired amplitude distribution of the output beam).
    error_threshold : float, optional
        The error threshold. The default is 1e-3.
    iterations : int
        Number of iterations.

    Returns
    -------
    array_like
        The reconstructed pattern.

    """
    # Initialize the reconstruction with random numbers.
    hologram = np.random.random(source.shape) * 2 * np.pi

    source = source.astype(np.complex128)
    target = target.astype(np.complex128)

    A = np.zeros_like(hologram).astype(np.complex128)
    B = np.zeros_like(hologram).astype(np.complex128)
    C = np.zeros_like(hologram).astype(np.complex128)
    D = np.zeros_like(hologram).astype(np.complex128)

    target_amp = np.sqrt(np.abs(target))
    source_amp = np.sqrt(np.abs(source))

    A = target_amp * np.exp(-1j * hologram)
    A = np.fft.ifft2(A)

    # Iterate until the error is below the threshold.
    for i in range(max_iterations):
        
        

        # Calculate the hologram.
        B = source_amp * np.exp(1j * np.angle(A))
        C = np.fft.fft2(B)
        D = target_amp * np.exp(1j * np.angle(C))
        A = np.fft.ifft2(D)

        # Calculate the error.
        error = np.sum(np.abs(np.abs(A) - target_amp)) / np.sum(target_amp)

        if error < error_threshold:
            break

        
    print(f"Error: {error:.3f} (iterations: {i}")

    return np.angle(A)


# %%
import skimage 
import skimage.draw
import matplotlib.pyplot as plt

input_beam = np.ones((IMAGE_SIZE, IMAGE_SIZE))
target_beam = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

# draw circle at (32, 32)

rc, cc = skimage.draw.disk((32, 32), 2, shape=target_beam.shape)
target_beam[rc, cc] = 4000
rc, cc = skimage.draw.disk((48, 35), 2, shape=target_beam.shape)
target_beam[rc, cc] = 4000
plt.imshow(target_beam)
plt.show()
# %%

hologram = gerchberg_saxton_phase_retrieval(input_beam, target_beam, max_iterations=100)
phase_hologram = np.exp(1j * hologram)
plt.imshow(hologram)
plt.show()

plt.imshow(np.abs(np.fft.fft2(phase_hologram)))
plt.colorbar()
# %%
