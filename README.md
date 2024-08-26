# Inteligencia Artificial (IA)

## ¿Que es IA?

**La inteligencia artificial, o IA, es tecnología que permite que las computadoras simulen la inteligencia humana y las capacidades humanas de resolución de problemas.**

![image](https://github.com/user-attachments/assets/3896efb0-7f9a-4ff0-ab94-a37a987e3d4c) 


Por sí sola o combinada con otras tecnologías (por ejemplo, sensores, geolocalización, robótica), la IA puede realizar tareas que de otro modo requerirían inteligencia o intervención humana. Los asistentes digitales, la guía por GPS, los vehículos autónomos y las herramientas de inteligencia artificial generativa (como Chat GPT de Open AI) son solo algunos ejemplos de inteligencia artificial en las noticias diarias y en nuestra vida cotidiana.

Como campo de la informática, la inteligencia artificial abarca (y a menudo se menciona junto con) el aprendizaje automático y el aprendizaje profundo. Estas disciplinas implican el desarrollo de algoritmos de IA, modelados a partir de los procesos de toma de decisiones del cerebro humano, que pueden "aprender" de los datos disponibles y realizar clasificaciones o predicciones cada vez más precisas con el tiempo.

La inteligencia artificial ha pasado por muchos ciclos de exageración, pero incluso para los escépticos, el lanzamiento de ChatGPT parece marcar un punto de inflexión. La última vez que la IA generativa fue tan importante, los avances se produjeron en la visión por computadora, pero el salto se produce en el procesamiento de lenguaje natural (PLN). Hoy en día, la IA generativa puede aprender y sintetizar no solo el lenguaje humano sino también otros tipos de datos, como imágenes, vídeos, códigos de software e incluso estructuras moleculares.

## Tipos de IA

**La IA débil, también llamada IA estrecha o Inteligencia artificial estrecha (ANI), es una IA entrenada y enfocada para realizar tareas específicas.** La IA débil impulsa la mayor parte de la IA que nos rodea hoy. "Estrecho" podría ser un descriptor más preciso para este tipo de IA, ya que no tiene nada de débil; permite algunas aplicaciones muy potentes, como Siri de Apple, Alexa de Amazon, IBM Watson y vehículos autónomos.

**La IA robusta está conformada por la inteligencia artificial general (IAG) y la superinteligencia artificial (SIA).** La inteligencia artificial general (IAG), o la IA general, es una forma teórica de IA en la que una máquina tendría una inteligencia igual a la de los humanos; sería autoconsciente y tendría la capacidad de resolver problemas, aprender y planificar para el futuro. La superinteligencia artificial (SIA), también conocida como superinteligencia, superaría la inteligencia y la capacidad del cerebro humano.

# Machine Learning (ML)

![image](https://github.com/user-attachments/assets/286d805a-954b-43f0-9da8-b3eed67249d4)


## ¿Qué es el 'machine learning' y para qué sirve?

El *Machine Learning* es una disciplina del campo de la **Inteligencia Artificial** que, a través de algoritmos, **dota a los ordenadores de la capacidad de identificar patrones en datos masivos y elaborar predicciones (análisis predictivo)**. Este aprendizaje permite a los computadores realizar tareas específicas de forma autónoma, es decir, sin necesidad de ser programados.


El término se utilizó por primera vez en 1959. Sin embargo, **ha ganado relevancia en los últimos años debido al aumento de la capacidad de computación y al boom de los datos**. Las técnicas de aprendizaje automático son, de hecho, una parte fundamental del ***Big Data.***


## Distintos algoritmos de 'machine learning'

Los algoritmos de *Machine Learning* se dividen en tres categorías, siendo las dos primeras las más comunes:

- ***Aprendizaje supervisado:*** estos algoritmos cuentan con un aprendizaje previo basado en un sistema de etiquetas asociadas a unos datos que les permiten tomar decisiones o hacer predicciones. Un ejemplo es un detector de spam que etiqueta un e-mail como spam o no dependiendo de los patrones que ha aprendido del histórico de correos (remitente, relación texto/imágenes, palabras clave en el asunto, etc.).

- ***Aprendizaje no supervisado:*** estos algoritmos no cuentan con un conocimiento previo. Se enfrentan al caos de datos con el objetivo de encontrar patrones que permitan organizarlos de alguna manera. Por ejemplo, en el campo del marketing se utilizan para extraer patrones de datos masivos provenientes de las redes sociales y crear campañas de publicidad altamente segmentadas.

- ***Aprendizaje por refuerzo:*** su objetivo es que un algoritmo aprenda a partir de la propia experiencia. Esto es, que sea capaz de tomar la mejor decisión ante diferentes situaciones de acuerdo a un proceso de prueba y error en el que se recompensan las decisiones correctas. En la actualidad se está utilizando para posibilitar el reconocimiento facial, hacer diagnósticos médicos o clasificar secuencias de ADN.

# ¿Por qué usar Python para Machine Learning?

![image](https://github.com/user-attachments/assets/c2685dbf-71ec-4279-a177-40e447e02552)


Python se ha convertido en uno de los lenguajes de programación más populares para el desarrollo de aplicaciones de Machine Learning. Esto se debe a una combinación de factores, incluyendo sus características, facilidad de uso y el amplio ecosistema de herramientas y librerías de Python para análisis de datos y visualización.

En cuanto a sus características para el desarrollo de Machine Learning, Python es un lenguaje interpretado y de alto nivel que cuenta con una sintaxis clara y concisa. Esto hace que sea fácil de aprender y utilizar, lo que lo convierte en una opción popular tanto para desarrolladores experimentados como para principiantes.

Además, Python cuenta con una amplia variedad de librerías de Machine Learning de código abierto, como Scikit-learn, TensorFlow y Keras, que facilitan la creación de modelos de Machine Learning de alta calidad con una variedad de algoritmos y técnicas de aprendizaje automático.

## Herramientas básicas de Python para Machine Learning

Existen muchas herramientas y librerías básicas útiles para el desarrollo de aplicaciones de Machine Learning en Python. A continuación, vamos a hablar de algunas de las más comunes.

NumPy es una librería de Python que se utiliza para trabajar con arrays multidimensionales y matrices, y proporciona una gran cantidad de funciones matemáticas para operar con ellos. Es una herramienta básica de Python para el procesamiento de datos numéricos y se utiliza ampliamente en la creación de modelos de Machine Learning. Ejemplo de uso:

~~~
import numpy as np
# crear un array de 3x3 con números aleatorios
arr = np.random.rand(3, 3)

# calcular la media de todos los elementos en el array
mean = np.mean(arr)

# imprimir el array y la media
print("Array:\n", arr)
~~~

Pandas es una librería de Python que se utiliza para la manipulación y análisis de datos en Python. Es útil para la limpieza de datos, la unión de conjuntos de datos, el filtrado y la agregación de datos, y la transformación de datos para su uso en modelos de Machine Learning. Ejemplo de uso:

~~~
import pandas as pd

# crear un DataFrame con dos columnas de números aleatorios
df = pd.DataFrame({'columna_1': [1, 2, 3, 4], 'columna_2': [5, 6, 7, 8]})

# agregar una nueva columna calculada como la suma de las otras dos columnas
df['columna_3'] = df['columna_1'] + df['columna_2']

# imprimir el DataFrame
print(df)
print("Media:", mean)
~~~

Matplotlib es una librería de Python que se utiliza para crear visualizaciones y gráficos de datos en Python. Es útil para visualizar conjuntos de datos y patrones de datos para ayudar a comprender los datos y ajustar los modelos de Machine Learning. Ejemplo de uso:

~~~
import matplotlib.pyplot as plt

# crear un array de 100 números aleatorios
arr = np.random.rand(100)

# crear un gráfico de línea con los números aleatorios
plt.plot(arr)

# agregar títulos y etiquetas al gráfico
plt.title("Gráfico de línea")
plt.xlabel("X")
plt.ylabel("Y")

# mostrar el gráfico
plt.show()
~~~

SciPy es una librería de Python que se utiliza para el procesamiento de señales, la optimización y la resolución de problemas numéricos. Proporciona una gran cantidad de funciones y herramientas para la creación de modelos de Machine Learning y la resolución de problemas numéricos. Ejemplo de uso:

~~~
import scipy

# definir una función para la optimización
def func(x):
    return x**2 + 10*np.sin(x)

# encontrar el mínimo de la función
result = scipy.optimize.minimize(func, 0)

# imprimir el resultado de la optimización
print(result)
~~~

## Librerías especializadas de Python en Machine Learning

Además de las herramientas básicas de Python para el desarrollo de aplicaciones de Machine Learning, también existen librerías especializadas que proporcionan herramientas y funcionalidades específicas para el entrenamiento y la implementación de modelos de Machine Learning. Aquí comentaremos algunas de las más populares.

Scikit-Learn es una librería de aprendizaje automático de Python que se utiliza ampliamente en la industria. Proporciona herramientas y funcionalidades para la selección de características, la validación, la selección y la implementación de modelos de Machine Learning en Python. En el siguiente ejemplo de uso vamos a mostrar cómo construir un árbol de decisión para predecir el tipo de flor del dataset iris:

~~~
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# cargar el conjunto de datos iris
iris = load_iris()

# dividir el conjunto de datos en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# crear un clasificador de árbol de decisión
clf = DecisionTreeClassifier()

# entrenar el clasificador con los datos de entrenamiento
clf.fit(X_train, y_train)

# evaluar el clasificador con los datos de prueba
score = clf.score(X_test, y_test)

# imprimir la precisión del clasificador
print("Precisión del clasificador:", score)
~~~

TensorFlow es una librería de aprendizaje automático de código abierto desarrollada por Google. Se utiliza ampliamente para la creación de modelos de Deep Learning y para la implementación de redes neuronales. En el siguiente ejemplo de uso veremos cómo construir una red neuronal con Tensorflow:

~~~
import tensorflow as tf

# crear una red neuronal con dos capas ocultas
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compilar el modelo con una función de pérdida y un optimizador
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# entrenar el modelo con un conjunto de datos de entrenamiento
model.fit(x_train, y_train, epochs=10, batch_size=32)

# evaluar el modelo con un conjunto de datos de prueba
score = model.evaluate(x_test, y_test, batch_size=32)

# imprimir la precisión del modelo
print("Precisión del modelo:", score[1])
~~~

Keras es una librería de Deep Learning de alto nivel que se ejecuta sobre TensorFlow y proporciona una interfaz de programación de aplicaciones (API) fácil de usar para la creación de modelos de redes neuronales. En el siguiente ejemplo de uso veremos como crear una red neuronal, en esta ocasión con Keras:

~~~
import keras
from keras.models import Sequential
from keras.layers import Dense

# crear una red neuronal con dos capas ocultas
model = Sequential()
model.add(Dense(64, input_dim=100, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compilar el modelo con una función de pérdida y un optimizador
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# entrenar el modelo con un conjunto de datos de entrenamiento
model.fit(x_train, y_train, epochs=10, batch_size=32)

# evaluar el modelo con un conjunto de datos de prueba
score = model.evaluate(x_test, y_test, batch_size=32)

# imprimir la precisión del modelo
print("Precisión del modelo:", score[1])
~~~

PyTorch es una librería de aprendizaje automático de código abierto desarrollada por Facebook. Se utiliza ampliamente para la creación de modelos de Deep Learning y para la implementación de redes neuronales. De nuevo vamos a mostrar cómo se crearía una red neuronal, en esta ocasión utilizando esta librería.

~~~
import torch
import torch.nn as nn
import torch.optim as optim

# crear una red neuronal con dos capas ocultas
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.softmax(self.fc3(x), dim=1)
        return x

net = Net()

# definir una función de pérdida y un optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# entrenar el modelo con un conjunto de datos de entrenamiento
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print("Epoch: %d, Loss: %.3f" % (epoch+1, running_loss/len(trainloader)))

# evaluar el modelo con un conjunto de datos de prueba
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Precisión del modelo: %d %%" % (100 * correct / total))
~~~

## Recursos útiles para Machine Learning en Python

Además de las herramientas y librerías de Python para Machine Learning, existen también recursos útiles que pueden facilitar el trabajo en este campo. A continuación, mencionaremos dos de ellos: Jupyter Notebook y Kaggle.

Jupyter Notebook es una aplicación web de código abierto que permite crear y compartir documentos que contienen código, texto explicativo, visualizaciones y otros elementos.

Es una herramienta muy útil para la exploración y visualización de datos, así como para la creación de prototipos y la experimentación con algoritmos de Machine Learning.

Jupyter se integra perfectamente con Python y es compatible con muchas de las librerías de Machine Learning más populares. Además, permite la ejecución de código de manera interactiva, lo que facilita el proceso de desarrollo y depuración.

Kaggle es una plataforma en línea que permite a los científicos de datos y desarrolladores de Machine Learning competir en desafíos y proyectos para resolver problemas del mundo real.

