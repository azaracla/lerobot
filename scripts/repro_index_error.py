
import torch
import math

def simulate_lerobot_index_calculation(ts, fps, num_frames):
    """Simulate the logic in src/lerobot/datasets/video_utils.py:345"""
    index = round(ts * fps)
    print(f"Timestamp: {ts}, FPS: {fps}, Calculated Index: {index}, Total Frames: {num_frames}")
    if index >= num_frames:
        print(f"BUG REPRODUCED: Index {index} is out of bounds for {num_frames} frames!")
        return True
    return False

# Dans votre log: Index=40482, Total=40482
# Si la vidéo fait 40482 frames à 30fps (hypothèse standard), 
# la durée totale est 40482 / 30 = 1349.4s

fps = 30.0 # On suppose 30fps car c'est courant
num_frames = 40482
# Un timestamp très proche de la fin, par exemple le timestamp exact de la dernière frame + un chouïa
last_frame_ts = (num_frames - 1) / fps # 1349.3666...
problematic_ts = last_frame_ts + 0.00001 # Un tout petit peu plus, mais tjs < durée totale

print("Testing with hypothetical 30fps video:")
simulate_lerobot_index_calculation(problematic_ts, fps, num_frames)

# Test avec le FPS exact qui causerait l'erreur pour un timestamp donné
# Si ts = 1024.0 et index = 40482, alors fps = 40482 / 1024 = 39.533203125
ts_at_error = 1024.0 # Hypothèse basée sur vos logs (step 20k, ~1000s)
inferred_fps = num_frames / ts_at_error
print(f"\nTesting with inferred FPS ({inferred_fps}):")
simulate_lerobot_index_calculation(ts_at_error, inferred_fps, num_frames)

# Reproduction avec un flottant qui arrondit au dessus
# 0.5 s'arrondit à 1 en Python (round(0.5) == 0 en Python 3? Non, round(0.5) est 0, round(1.5) est 2. 
# En fait round() en Python 3 arrondit vers l'entier pair le plus proche pour les .5
print(f"\nPython round(0.5): {round(0.5)}")
print(f"Python round(1.5): {round(1.5)}")
print(f"Python round(0.5000001): {round(0.5000001)}")
