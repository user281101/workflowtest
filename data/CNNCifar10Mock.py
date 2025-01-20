#2 Datenanalyse und Vorbereitung
# Library for plotting the images and the loss function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
import keras
import os

#tensorflow mit gpu installiert
print("GPU verfügbar:", tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    print("Training läuft auf GPU!")
else:
    print("Keine GPU verfügbar. Training läuft auf CPU.")
##
#zuerst testdaten aufsplitten dann skalieren um datenlecks zu vermeiden 

#testdatenaufteilung funktioniert
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print('Training set shape:', x_train.shape) #anzeigen der datenverteilung für die übersicht
print('Test set shape:', x_test.shape)

#visualisierung der Klassenverteilung zur Veranschaulichung
klassen = ['Flugzeug','Auto','Vogel','Katze','Reh','Hund','Frosch','Pferd','Schiff','LKW']
classes, counts = np.unique(y_train, return_counts=True)
plt.barh(klassen, counts)
plt.title('klassenverteilung im training set')

#Visualisierungen der Bilder(Sollte am pro Klasse ein Exemplar sein) funktioniert
def visualize_cifar10_images(images, labels, class_names, num_images): 
    plt.figure(figsize=(10, 5)) 
    for i in range(num_images): 
        plt.subplot(2, 5, i + 1) 
        plt.imshow(images[i]) 
        plt.title(class_names[labels[i][0]]) 
        plt.axis('off') 
    plt.tight_layout() 
    plt.show()

visualize_cifar10_images(x_train, y_train, klassen, num_images=10)

# Datenskalierung funktioniert
x_train = x_train / 255.0
x_test = x_test / 255.0


#one hot encoding labels für nicht nötig gewesen dank SparseCategoricalCrossentropy
#was macht SparseCategoricalCrossentropy?
#validation trainingset zum schauen wie die predictions sind ohne testset anzurühren
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state = 42) #randomstate Datenaufteilung reproduzierbar
#bisher:Datensatz laden, training_split ,visualisierung der Bilder und der Testdaten-Aufteilung,Datenskalierung, One hot encoding und validation Trainingsset erfolgreich

#3.modellaufbau und training
from keras import Sequential,layers, optimizers
from keras import regularizers

INPUT_SHAPE = (32, 32, 3) 
KERNEL_SIZE = (3, 3)

#relu verwendet um nichtlinearität zu erzeugen für cnn.
#ReLU ist effizient und verhindert das Verschwinden des Gradienten, wodurch tiefe Netzwerke besser trainiert werden können.
#convolutional layer

#Hyperparameter: Komplexeres netz, anzahl neuronen pro schicht erhöht(besser für komplexe Beziehungen)

model = Sequential([
    #layer1
    keras.layers.Conv2D(32, (3, 3), padding= 'same',activation='relu', input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), padding = 'same',activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),

    #layer2
    keras.layers.Conv2D(64, (3, 3), padding = 'same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), padding = 'same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),   #2x2 Matrix zur verkleinerung, spart an rechenleistung
    keras.layers.Dropout(0.5),
    
    #layer3
    keras.layers.Conv2D(128, (3, 3), padding = 'same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), padding = 'same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.5),

    #layer4
    keras.layers.Flatten(), #Konvertiert die 2D-Ausgabe (Feature Maps) der vorherigen Schicht in einen eindimensionalen Vektor. Dies ist notwendig, um die Daten an vollständig verbundene Schichten (Dense Layers) weiterzugeben.
    keras.layers.Dense(128, activation='relu'),         #64 Neuronen in der versteckten Schicht
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),                    #dropout verbessert die test accuracy um 0.17 bei unserem code, hilft gegen overfitting entfernt 50% der neuronen, Teil von 5. Hyperparameter
    keras.layers.Dense(10, activation='softmax')  # 10 Klassen für CIFAR-10, softmax 
])

# Compiling the model 
#SparseCategoricalCrossentropy für berechnung des Loss, warum SparseCategoricalCrossentropy und nicht CategoricalCrossentropy?
#learning rate mit einbauen für Hyperparameter tuning
model.compile(optimizer = 'adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])  #Logits = vorhergesagte Wahrscheinlichkeiten
#modellausgabe sind bereits Wahrscheinlichkeiten 

# Training the model for 10 epochs, validierungsdaten genutzt um Datenlecks zu vermeiden
#loss minimum bei eopche 4, early stoppage toleriert 3 eopchen ohne verbesserung danach setback auf den niedrigsten loss wert
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_valid, y_valid), callbacks=[early_stopping])

#Hyperparameter: batchsize und epochen erhöht = höhere accuracy

#4 Evaluieren und Konfusionsmatrix
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

#modell evaluieren
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.2f}")

# Vorhersagen generieren
y_pred = np.argmax(model.predict(x_test), axis=-1)


def plot_confusion_matrix(model, x_test, y_test, save_dir='data/plots'):
    """Plot and save the confusion matrix."""
# Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    plt.show()
#konfusionsmatrix , zeigen wie gut das Modell die Klassen unterscheidet
ConfusionMatrixDisplay.from_predictions(y_test.flatten(), y_pred, display_labels=klassen)
plt.show()

# Klassifikationsbericht für detaillierte Übersicht zur Modelleistung pro Klasse
print("y_test:", y_test)
print("y_pred:", y_pred)
print("Klassifikationsbericht:")
print(classification_report(y_test, y_pred, target_names=klassen))

#6 visualisieren der Ergebnisse
#Darstellung der Trainings- und Validierungsverluste sowie der Accuracy-Werte über die Trainingsperioden, speichert die plots als png datei
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoche')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig("training_history.png")
plt.close()

#Darstellung falsch klassifizierter Bilder , von chatgpt
misclassified_indices = np.where(y_test.flatten() != y_pred)[0]
plt.figure(figsize=(12, 12))
for i, index in enumerate(misclassified_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[index])
    plt.title(f"True: {klassen[y_test[index][0]]}, Pred: {klassen[y_pred[index]]}")
    plt.axis('off')
plt.tight_layout()
plt.show() 
