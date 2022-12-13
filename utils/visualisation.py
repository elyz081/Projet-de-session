
import numpy as np
import matplotlib.pyplot as plt

def visualize(ac_before,ac_after):
    """
    param ac_before: Dict of model's accuracies before data scalling
    param ac_after:  Dict of model's accuracies after data scalling
    """

    models= ac_before.keys()

    accuracy_before = [ac_before[model] for model in models]
    accuracy_after = [ac_after[model] for model in models]

    
    largeur = 0.4  
    plt.subplots(figsize=(10,6))
    plt.margins(0.06, 0.1) 
    bar1 = np.arange(len(models))
    bar2 = [i+largeur for i in bar1]

    plt.bar(bar1, accuracy_before, largeur, label='Accuracy before', color='coral')
    plt.bar(bar2, accuracy_after, largeur, label='Accuracy after', color='teal')

    plt.xticks(bar1+largeur/2, models, rotation=50)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.legend(loc=9, ncol=2)

    plt.title(f"Accuracy of the chosen models before and after hyperparameters search")
    plt.show()
 