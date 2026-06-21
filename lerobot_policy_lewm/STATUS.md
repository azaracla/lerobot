Résumé de la session & statut actuel
Objectif initial

Intégrer LeWM (JEPA world model avec planification par CEM) comme plugin dans LeRobot (Hugging Face) et l’entraîner sur PushT pour atteindre un bon taux de succès en évaluation MPC.
Ce qui a été accompli

    Plugin lerobot_policy_lewm créé et testé (56 tests passés).

    Compatibilité lerobot-train atteinte après corrections :

        reward_delta_indices retourne None (au lieu de []).

        ImagePreprocessorStep.observation() correctement nommé.

        Détection des clés "image" dans les observations.

        Import corrigés (normalize_processor, device_processor).

        Ajout des steps RenameObservations et Device pour l’évaluation.

        Gestion des tenseurs sans dimension temporelle (unsqueeze).

        Vérification de présence de "action" dans le batch.

    lerobot-eval fonctionne avec le plugin (mais sans goal conditioning intégré).

    Git : branche local-dev poussée sur le fork (azaracla/lerobot) avec tous les commits.

    Compréhension des bugs :

        AdaLN gates à zéro : l’init to_out et mlp[-2] à zéro bloquait les gradients (corrigé en suivant l’original LeWM).

        Frames trop proches : avec frameskip=1 (0.1 s entre frames), la prédiction est triviale et la loss s’effondre. L’original utilise frameskip=5 (0.5 s).

        Loss non nulle après correction : elle descend bien mais très vite (possiblement due à la simplicité de PushT).

    Entraînements :

        25K steps → loss 0.000 mais évaluation 0 %.

        100K avec batch 32 (tentative) → interrompu suite au diagnostic.

        10K avec frameskip=5 → gates s’ouvrent, coût CEM varie (variance 1e-5), mais évaluation toujours 0 % à 2500 steps.

Statut actuel

    Dernier entraînement : 10K steps avec frameskip=5, batch 32, lancé en arrière-plan (lewm_pusht_10k_skip5).

        Checkpoint 2500 disponible.

        Les gates AdaLN ont une norme ~0.00085 (non nulles).

        Les coûts CEM sont différents selon les actions (range ~1.6e-5), ce qui montre que le modèle utilise les actions.

        Évaluation MPC sur 2500 steps : toujours 0 % de succès sur 5 épisodes (actions semblent rester mauvaises).

    Blocage principal :

        Le modèle n’est pas encore assez entraîné pour qu’un CEM de 200 échantillons × 15 itérations trouve des actions optimales.

        Le goal conditioning n’est pas pleinement intégré dans l’évaluation standard (lerobot-eval). Le script d’évaluation maison utilise swm env qui supporte _set_goal_state, mais le coût reste basé sur la dernière frame observée, pas sur un but explicite (image de goal). Cela peut limiter la capacité du CEM.

        Hyperparamètres CEM : peut-être trop faibles (200×15) comparé au papier (300×30).

Prochaines étapes & recommandations

    Laisser tourner l’entraînement 10K avec frameskip=5 jusqu’à la fin (ou plus, 25K/50K).

    Tester l’évaluation avec plus d’échantillons CEM (ex. 500×30) pour voir si le modèle 10K peut mieux planifier.

    Améliorer le goal conditioning :

        Utiliser l’environnement swm/PushT-v1 dans l’évaluation standard, ou passer explicitement une image de goal.

        Vérifier que le coût CEM utilise bien l’embedding de goal et non une frame vide.

    Valider les gradients : s’assurer que la loss n’est pas toujours quasi nulle (utiliser des métriques comme la variance des embeddings prédits).

    À plus long terme : reproduire plus fidèlement l’original LeWM (batch=128, plus de steps, etc.) si les ressources le permettent.
