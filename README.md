#  Face-Emotions — Reconnaissance des émotions faciales (Flask + Keras + MediaPipe)

##  Introduction

Les **expressions faciales** sont un langage universel essentiel à la communication non verbale.  
L’analyse automatique des émotions ouvre la voie à des **interactions homme–machine plus naturelles**, à des outils de **santé mentale**, et à des **systèmes de surveillance intelligents**.

Grâce aux progrès de l’**intelligence artificielle** et des **réseaux de neurones convolutifs (CNN)**, il est désormais possible de concevoir des systèmes capables de **détecter et classifier les émotions faciales en temps réel** avec une précision raisonnable.

---

##  Problématique

Malgré les avancées, la reconnaissance d’émotions faciales **en conditions réelles** reste un défi :  
variabilité des visages, conditions d’éclairage, postures, et diversité des expressions.

**Problème central :**  
> Comment concevoir un système automatisé, fiable, léger et rapide, capable d’identifier correctement les émotions humaines à partir d’images de visages dans des environnements réels ?

---

##  Objectifs du projet

- Développer un **modèle CNN** de classification d’émotions sur des images de visages en niveau de gris.  
- Intégrer un **module de détection faciale précis et rapide**.  
- Concevoir une **interface web interactive** avec **Flask**, capable de traiter :
  - des images statiques (upload),
  - et des flux vidéo en temps réel (webcam).

---

##  Méthodologie générale

L’approche repose sur :

- **Dataset** : [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) (35 887 images 48×48 px, 7 émotions)
- **Prétraitement** : OpenCV (conversion, redimensionnement, normalisation, encodage)
- **Détection faciale** : MediaPipe (rapide, précise, temps réel)
- **Modélisation** : CNN personnalisé (Keras)
- **Déploiement** : Application Flask intégrant les modèles statiques et temps réel

---

##  Données et prétraitement

###  Jeu de données FER2013
- 35 887 images en **niveaux de gris 48×48**
- 7 émotions : *colère, dégoût, peur, joie, neutre, tristesse, surprise*
- Données déséquilibrées → nécessité d’**augmentation de données**

###  Prétraitement
- Chargement et conversion des pixels
- Normalisation entre 0 et 1
- Encodage **one-hot**
- **Data augmentation** : flip horizontal, zoom, rotation, translation

---

##  Détection des visages (MediaPipe)

### Rôle
La classification nécessite une **région centrée sur le visage**.  
MediaPipe isole automatiquement cette zone, même avec plusieurs visages ou un fond complexe.

### Étapes du pipeline
1. Lecture image (OpenCV)  
2. Conversion BGR → RGB  
3. Détection du visage (bounding box)  
4. Découpage et redimensionnement (48×48)  
5. Normalisation → passage au CNN

---

##  Modélisation (CNN personnalisé)

### Architecture
- 4× **Convolution 3×3 (ReLU + padding=same)**
- **MaxPooling2D** après chaque bloc
- **Dropout (0.25–0.5)** pour éviter le surapprentissage
- **Flatten** → **Dense** → **Softmax (7 neurones)**

### Paramètres d’entraînement
- Optimiseur : **Adam**
- Perte : **categorical crossentropy**
- Batch size : 64  
- Epochs : 30–50 (avec **early stopping**)
- Scheduler de **learning rate**

---

##  Évaluation du modèle

- **Accuracy validation** : ~59.5 %
- **Meilleure émotion reconnue** : *Joie*
- **Moins bien reconnues** : *Peur*, *Dégoût*

###  Matrice de confusion
- Bonnes distinctions : *Joie*, *Neutre*, *Tristesse*  
- Confusions fréquentes : *Peur ↔ Surprise*, *Colère ↔ Dégoût*

### Limites
- Images petites (48×48)
- Dataset peu diversifié
- Architecture CNN simple (pas de transfert learning)

---

##  Application Flask

### Structure de l’application

### Routes principales
| Route | Fonction |
|-------|-----------|
| `/` | Page d’accueil |
| `/upload` | Upload d’image + affichage du résultat |
| `/video` | Mode webcam |
| `/video_feed` | Flux MJPEG temps réel |

### Traitement d’une image
1. Lecture du fichier
2. Détection faciale via MediaPipe
3. Redimensionnement & normalisation
4. Prédiction du CNN
5. Affichage de l’émotion et du score de confiance

---

##  Tests & performances réelles

- Fonctionnement fluide (≈ 20–25 FPS)
- Latence faible (< 1 s)
- Bonne détection en lumière stable
- Précision variable selon luminosité ou expressions atypiques

### Résultats globaux
- Accuracy d’entraînement : **63 %**
- Accuracy de validation : **60 %**
- Perte de validation stable ≈ 1.1 à partir de l’époque 20
- F1-score global ≈ 0.58

| Émotion | F1-score |
|----------|----------|
|  Happy | 0.798 |
|  Surprise | 0.717 |
|  Fear | 0.346 |
|  Angry | 0.52 |
|  Neutral | 0.65 |

---

##  Lancement rapide

### Installation
```bash
git clone https://github.com/8Abdoul/face-emotions.git
cd face-emotions

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
