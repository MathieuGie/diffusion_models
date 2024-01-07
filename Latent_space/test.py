import os
import shutil

# Chemin du dossier principal
main_folder_path = "/mnt/d/Hanna/Polytechnique/Deep_Learning/dataset/animals"

# Parcourir tous les sous-dossiers et fichiers
for subdir, dirs, files in os.walk(main_folder_path):
    for file in files:
        # Construire le chemin complet du fichier
        file_path = os.path.join(subdir, file)
        
        # Vérifier si le fichier n'est pas déjà dans le dossier principal
        if subdir != main_folder_path:
            # Construire le chemin de destination dans le dossier principal
            destination_path = os.path.join(main_folder_path, file)

            # Déplacer le fichier
            shutil.move(file_path, destination_path)
            print(f"File {file} moved to {main_folder_path}")

# Supprimer les sous-dossiers vides
for subdir in os.listdir(main_folder_path):
    subdir_path = os.path.join(main_folder_path, subdir)
    if os.path.isdir(subdir_path):
        try:
            os.rmdir(subdir_path)  # Ceci supprimera le sous-dossier s'il est vide
            print(f"Empty folder {subdir} removed")
        except OSError as e:
            print(f"Folder {subdir} is not empty and has not been removed. Error: {e}")

print("All files have been moved to the main folder and empty folders have been removed.")
