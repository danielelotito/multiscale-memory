import imageio
import matplotlib.pyplot as plt
import numpy as np


import os

from setup_matplotlib_style import setup_matplotlib_style

def dynamic_visualizations(simulation_dict, patterns, Hebb, cfg):
    setup_matplotlib_style()
    if cfg.max_size < cfg.Js_size:
        return 0
    if cfg.visualizations.dynamic:
        # Extract data from simulation_dict
        times = np.array(simulation_dict["times"])
        sigmas = np.array(simulation_dict["sigmas"])
        Js = np.array(simulation_dict["Js"])
        fields = np.array(simulation_dict["fields"])
        # WARNING: Plots involving Tras and Xis are not implemented yet
        # WARNING: check if sigmas, Js have same shape as before
        if "neural_activities_gif" in cfg.visualizations.dynamic:
            create_neural_activities_gif(times, sigmas, cfg)
        if "synaptic_weights_gif" in cfg.visualizations.dynamic:
            create_synaptic_weights_gif(times, Js, cfg)
        if "synaptic_hebb_difference_gif" in cfg.visualizations.dynamic:
            create_synaptic_Hebb_difference_gif(times, Js, Hebb, cfg)
        if "phase_space_gif" in cfg.visualizations.dynamic:
            create_phase_space_gif(times, sigmas, Js, cfg)


def create_neural_activities_gif(times, sigmas, cfg):
    fig, ax = plt.subplots(figsize=(10, 6))
    images = []

    # Calculate global min and max for consistent colorbar
    vmin = np.min(sigmas)
    vmax = np.max(sigmas)
    # Create a colorbar that will be consistent for all frames
    im = ax.imshow(
        sigmas[0].reshape(1, -1), aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Neural Activities", rotation=270, labelpad=15)

    for i in range(len(times)):
        ax.clear()
        ax.imshow(sigmas[i].reshape(1, -1), aspect="auto", cmap="coolwarm")
        ax.set_title(f"Neural Activities at t={times[i]:.2f}")
        ax.set_yticks([])
        ax.set_xlabel("Neuron index")

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

    if cfg.save.enabled:
        imageio.mimsave(
            os.path.join(
                cfg.save.folders.dynamic_vis, cfg.name + "_neural_activities.gif"
            ),
            images,
            fps=10,
        )
    plt.close(fig)


def create_synaptic_weights_gif(times, Js, cfg):
    fig, ax = plt.subplots(figsize=(10, 6))
    images = []

    # Calculate global min and max for consistent colorbar
    vmin = min(np.min(J) for J in Js)
    vmax = max(np.max(J) for J in Js)
    # Create a colorbar that will be consistent for all frames
    im = ax.imshow(Js[0], aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Synaptic Weights", rotation=270, labelpad=15)

    for i in range(len(times)):
        ax.clear()
        ax.imshow(Js[i], aspect="auto", cmap="viridis")
        ax.set_title(f"Synaptic Weights at t={times[i]:.2f}")
        ax.set_xlabel("Neuron j")
        ax.set_ylabel("Neuron i")

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

    if cfg.save.enabled:
        imageio.mimsave(
            os.path.join(
                cfg.save.folders.dynamic_vis, cfg.name + "_synaptic_weights.gif"
            ),
            images,
            fps=10,
        )
    plt.close(fig)


def create_synaptic_Hebb_difference_gif(times, Js, Hebb, cfg):
    fig, ax = plt.subplots(figsize=(10, 6))
    images = []

    # Calculate global min and max for consistent colorbar
    vmin = min(np.min(abs(J - Hebb)) for J in Js)
    vmax = max(np.max(abs(J - Hebb)) for J in Js)
    # Create a colorbar that will be consistent for all frames
    im = ax.imshow(
        abs(Js[0] - Hebb), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Absolute Difference", rotation=270, labelpad=15)

    for i in range(len(times)):
        ax.clear()
        ax.imshow(abs(Js[i] - Hebb), aspect="auto", cmap="viridis")
        ax.set_title(f"|Synaptic Weights - Hebb| at t={times[i]:.2f}")
        ax.set_xlabel("Neuron j")
        ax.set_ylabel("Neuron i")

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

    if cfg.save.enabled:
        imageio.mimsave(
            os.path.join(
                cfg.save.folders.dynamic_vis, cfg.name + "_synaptic_hebb_difference.gif"
            ),
            images,
            fps=10,
        )
    plt.close(fig)


def create_phase_space_gif(times, sigmas, Js, cfg):
    fig, ax = plt.subplots(figsize=(10, 6))
    images = []
    mean_sigmas = np.mean(sigmas, axis=1)
    mean_Js = np.mean(Js, axis=(1, 2))

    for i in range(len(times)):
        ax.clear()
        ax.plot(mean_sigmas[: i + 1], mean_Js[: i + 1])
        ax.set_xlim(mean_sigmas.min(), mean_sigmas.max())
        ax.set_ylim(mean_Js.min(), mean_Js.max())
        ax.set_title(f"Phase Space Trajectory at t={times[i]:.2f}")
        ax.set_xlabel("Mean neural activity")
        ax.set_ylabel("Mean synaptic weight")

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

    if cfg.save.enabled:
        imageio.mimsave(
            os.path.join(cfg.save.folders.dynamic_vis, cfg.name + "_phase_space.gif"),
            images,
            fps=10,
        )
    plt.close(fig)
