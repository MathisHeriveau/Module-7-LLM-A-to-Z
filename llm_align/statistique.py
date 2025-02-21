import matplotlib.pyplot as plt
import numpy as np
import re

# Charger le fichier log
# log_file_path = "C:/Users/MathisHERIVEAU/Documents/Apprentissage IA/Deep Learning/Module 7 - LLM - A to Z/llm_align/models/training_train_align.log"
log_file_path = "C:/Users/MathisHERIVEAU/Documents/Apprentissage IA/Deep Learning/Module 7 - LLM - A to Z/llm_align/models/training_knowledge.log"

# Lire le fichier et extraire les valeurs de Loss et Odds Ratio
iterations, losses, odds_ratios, epochs = [], [], [], []

with open(log_file_path, "r", encoding="utf-8") as file:
    for line in file:
        # match = re.search(r"Epoch: \[(\d+)/\d+\] Iteration: \[(\d+)/\d+\] Loss: ([\d\.]+) Odds Ratio: ([\d\.-]+)", line)
        match = re.search(r"Epoch: \[(\d+)/\d+\] Iteration: \[(\d+)/\d+\] Loss: ([\d\.]+)", line)
        if match:
            epoch = int(match.group(1))  # Numéro d'epoch
            iteration = int(match.group(2))  # Numéro d'itération
            loss = float(match.group(3))  # Valeur de la loss
            # odds_ratio = float(match.group(4))  # Odds Ratio
            
            epochs.append(epoch)
            iterations.append(iteration)
            losses.append(loss)
            # odds_ratios.append(odds_ratio)

# Appliquer un lissage avec une moyenne mobile (window=10)
def smooth_curve(data, weight=0.9):
    smoothed = []
    last = data[0]  # Premier point inchangé
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # Filtre exponentiel
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

smooth_losses = smooth_curve(losses, weight=0.99)
# smooth_odds_ratios = smooth_curve(odds_ratios, weight=0.99)

# Définir des couleurs par epoch
colors = ["blue", "green", "orange", "purple", "red", "cyan"]
unique_epochs = sorted(set(epochs))  # Obtenir la liste unique des epochs triés

plt.figure(figsize=(12, 5))

# Tracer le Loss avec couleur par epoch
plt.subplot(1, 2, 1)
for epoch in unique_epochs:
    mask = np.array(epochs) == epoch  # Sélectionner uniquement les points de cet epoch
    plt.plot(np.array(iterations)[mask], np.array(smooth_losses)[mask], label=f"Epoch {epoch}", color=colors[epoch % len(colors)])

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss (lissé)")
plt.legend()
plt.grid()

# Tracer l'Odds Ratio avec couleur par epoch
# plt.subplot(1, 2, 2)
# for epoch in unique_epochs:
#     mask = np.array(epochs) == epoch
#     plt.plot(np.array(iterations)[mask], np.array(smooth_odds_ratios)[mask], label=f"Epoch {epoch}", color=colors[epoch % len(colors)])

# plt.xlabel("Iterations")
# plt.ylabel("Odds Ratio")
# plt.title("Odds Ratio (lissé)")
# plt.legend()
# plt.grid()

plt.tight_layout()
plt.show()
