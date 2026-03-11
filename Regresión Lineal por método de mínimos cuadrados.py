#Recopilamos los datos para la variable independiente (x) y la variable dependiente (y)
#Generamos los datos de ejemplo por medio de dos arreglos de numpy
import numpy as np
import matplotlib.pyplot as plt
 
x = np.array ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
y = np.array ([2, 5, 7, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59])
n = len(x)

#Calculamos las medias de las variables x e y 
x_mean = 1/n * np.sum(x)
y_mean = 1/n * np.sum(y)

#Calculamos la covarianza entre x e y
S_xy = np.sum((x-x_mean)*(y - y_mean))

#Calculamos la varianza de x 
S_xx = np.sum ((x - x_mean)**2)

#Calculamos los parametros optimos de la recta y estructura de la ecuacion de la recta
m = S_xy / S_xx
b = y_mean - m * x_mean
y_pred = m*x + b

#Calculamos el coeficinete de determinación R^2
e = y - y_pred 
R_2 = 1 - (np.sum(e**2) / np.sum((y - y_mean)**2))
           
#Imprimimos los resultados de la ecuacion de la recta y el coeficiente de determinación R^2
print("La ecuacion final es y = ", f"{m:.4f}", "x + ", f"{b:.4f}")
print("El coeficiente de determinación R^2 es: ", f"{R_2:.3f}")

#Para demostrar el resultado, graficamos los datos y la recta de regresión
plt.scatter(x, y, color = "black", label = "Datos")
plt.plot(x, y_pred, color = "blue", label = "Recta de regresión")
plt.xlabel("Variable independiente (x)")
plt.ylabel("Variable dependiente (y)")
plt.title("Regresión Lineal por Métodos Mínimos Cuadrados")
plt.legend()
plt.show()
