import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

number_of_files = 15
for n in range(number_of_files):

  file = np.loadtxt(f"Dane/dane{n + 1}.txt")
  X = file[:, [0]]
  y = file[:, [1]]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)#podział danych na zbiór treningowy i testowy

  print(f"Plik: dane{n+1}.txt")
  #model liniowy y = ax + b
  macierz_liniowa = np.hstack([X_train, np.ones(X_train.shape)]) #macierz [x, 1]
  liniowe = np.linalg.pinv(macierz_liniowa) @ y_train #wyznaczenie współczynników wielomianu liniowego,pseudo wyznaczenie macierzy odwrotnej
  model_liniowy = liniowe[0] * X + liniowe[1] #wyznaczenie wyniku dla wszystkich punktów
  print("Współczynniki wielomianu liniowego: {}".format(liniowe))
  #błąd dla zbioru treningowego
  blad_liniowy_train = sum((y_train - (liniowe[0] * X_train + liniowe[1])) ** 2) / len(X_train)
  print("Błąd dla zbioru treningowego: {}".format(blad_liniowy_train))
  #błąd dla zbioru testowego
  blad_liniowy_test = sum((y_test - (liniowe[0] * X_test + liniowe[1])) ** 2) / len(X_test)
  print("Błąd dla zbioru testowego: {}".format(blad_liniowy_test))
  print('\n')

  #model nieliniowy y = w2 * x^3 + w1 * x^2 + w0
  macierz_nieliniowa = np.hstack([pow(X_train, 3), pow(X_train, 2), np.ones(X_train.shape)]) #macierz[x, 1]
  nieliniowe = np.linalg.pinv(macierz_nieliniowa) @ y_train #wyznaczenie współczynników wielomianu nieliniowego,pseudo wyznaczenie macierzy odwrotnej
  model_nieliniowy = nieliniowe[0] * pow(X, 3) + nieliniowe[1] * pow(X, 2) + nieliniowe[2] #wyznaczenie wyniku dla każdego punktu z zbioru treningowego
  print("Współczynniki wielomianu nieliniowego: {}".format(nieliniowe))
  #błąd dla zbioru treningowego
  blad_nieliniowy_train = sum((y_train - (nieliniowe[0] * pow(X_train, 3) + nieliniowe[1] * pow(X_train, 2) + nieliniowe[2])) ** 2) / len(X_train)
  print("Błąd dla zbioru treningowego: {}".format(blad_nieliniowy_train))
  #błąd dla zbioru testowego
  blad_nieliniowy_test = sum((y_test - (nieliniowe[0] * pow(X_test, 3) + nieliniowe[1] * pow(X_test, 2) + nieliniowe[2])) ** 2) / len(X_test)
  print("Błąd dla zbioru testowego: {}".format(blad_nieliniowy_test))
  print('\n')

  plt.plot(X_test, y_test, 'o')
  plt.plot(X_train, y_train, 'o')
  plt.plot(X, model_liniowy)
  plt.plot(X, model_nieliniowy)
  plt.show()