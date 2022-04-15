Pour lancer le programme, exécuter avec python la commande
    main.py
Les configurations de base sont algorithm = "A1", verbose = False, test = False 
Mais vous pouvez les modifier avant d'exécuter, voici les arguments 
    main.py --algorithm ["A1","A2","G"] --verbose --test

Dans le programme on vous demande de choisir un mot secret dans le dictionnaire ou de choisir la taille d'un mot aléatoire
Le programme exécute et affiche les resultats avant de se terminer.

Dans le mode test l'algorithme selectionné, va se lancer automatiquement pour des 10 mots au hasard de longueur 4 a 8 choisi aleatoirement,
pour 50 mots au total.
Les resultats de ces tests sont ensuite enregistré dans un fichier csv dans le dossier courant.

Nos résultats sont dans le dossier test