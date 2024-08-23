# Instrukcja obsługi

## gan_utils.py
Zawiera funkcje pomocnicze, potrzebne podczas treningu i wizualizacji wyników, jednak nie wchodzące bezpośrednio w skład modeli ani algorytmów uczenia

## generator_model.py
Kod tworzący model generatora. Funkcja create() zwraca obiekt-model

## discriminator_model.py
Analogicznie jak poprzedni, ale zwraca generator. Do funkcji create() można podać parametr, w zależności czy chcemy dostać model w wersji niemieckiej, czy nie

## train_gan_wasserstein.py
Zawiera algorytm uczenia sieci neuronowej. Ten plik należy uruchomić, aby rozpocząć uczenie

UWAGA: Kod jest przystosowany do wersji Tensorflow, która jest zainstalowana na serwerach Google colab. Może nie działać odpalony lokalnie

## scanner.py
Zmienia mapy w macierze na potrzeby GANa za pomocą funkcji.
scan_map(): tworzy tensor z całej mapy
debug_scan_map(): duplikuje mapę i nakłada na nią klasyfikację
serialize_map_submatrices() : dzieli mapę na kawałki  i ją serializuje

