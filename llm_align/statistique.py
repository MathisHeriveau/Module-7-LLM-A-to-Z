import matplotlib.pyplot as plt
import numpy as np
import re

log_file_path = "C:/Users/MathisHERIVEAU/Documents/Apprentissage IA/Deep Learning/Module 7 - LLM - A to Z/llm_align/models/training_knowledge_boost.log"

iterations, losses, odds_ratios, epochs = [], [], [], []
validation_iterations, validation_losses = [], []

val_loss_count = 1
with open(log_file_path, "r", encoding="utf-8") as file:
    for i, line in enumerate(file):  # On récupère maintenant aussi le numéro de la ligne (i)
        # Extraction des informations de training
        match = re.search(r"Epoch: \[(\d+)/\d+\] Iteration: \[(\d+)/\d+\] Loss: ([\d\.]+)", line)
        if match:
            epoch = int(match.group(1)) 
            iteration = int(match.group(2)) 
            loss = float(match.group(3))  
            
            epochs.append(epoch)
            iterations.append(iteration)
            losses.append(loss)

        # Extraction des informations de validation
        val_match = re.search(r"Epoch: \[(\d+)/\d+\] Validation Loss: ([\d\.]+)", line)
        if val_match:
            val_loss = float(val_match.group(2))
            
            # Utiliser le numéro de ligne pour déterminer l'itération de validation
            validation_iterations.append(i*50 - val_loss_count *50)  # 5000 itérations par ligne
            validation_losses.append(val_loss)
            val_loss_count += 1

def smooth_curve(data, weight=0.9):
    smoothed = []
    last = data[0]  
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point 
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

smooth_losses = smooth_curve(losses, weight=0.99)
colors = ["blue", "green", "orange", "purple", "red", "cyan"]
unique_epochs = sorted(set(epochs))  

# Plot des Losses lissés
plt.figure(figsize=(12, 5))

# Graphique 1 : Training Loss (perte d'entraînement)
plt.subplot(1, 2, 1)
for epoch in unique_epochs:
    mask = np.array(epochs) == epoch  
    plt.plot(np.array(iterations)[mask], np.array(smooth_losses)[mask], label=f"Epoch {epoch}", color=colors[epoch % len(colors)])

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss (lissé)")
plt.legend()
plt.grid()

# Graphique 2 : Validation Loss (perte de validation)
plt.subplot(1, 2, 2)
plt.plot(np.array(iterations), np.array(smooth_losses), label="Training Loss", color='blue')
plt.scatter(validation_iterations, validation_losses, color='red', label='Validation Loss', marker='x')

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
