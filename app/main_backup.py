import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import pickle

from PIL import Image
def main():
    st.title('Mushroom Classifier')
    st.write('Welcome to your categorical and photographical Mushroom Classifier')

    file= st.file_uploader('Please, upload your mushroom photo', type=['jpg','png'])
    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        resized_image = image.resize((48,48))
        img_array = np.array(resized_image) /255
        img_array = img_array.reshape(1,48,48,3)
        with open('..\\models\\cnn_model.pkl', 'rb') as file:
            model = pickle.load(file)

        predictions= model.predict(img_array)
        mushroom_classes = [' Amanita Vaginata Var - Vaginata', ' Amanita brunneitoxicaria', ' Amanita phalloides-',
                            ' Amanita princeps Corner - Bas', ' Chlorophyllum molybdites', ' Lactarius glaucescens',
                                ' Lentinus polychrous Berk', ' Lentinus squarrosulas Mont', ' Macrolepiota gracilenta',
                                ' Mycoamaranthus cambodgensis', ' Pleurotus pulmonarius', ' Schizophylllum commune', 
                                ' Scleroderma sinnamariense']
        poisonous_indicator = [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
        predicted_species = mushroom_classes[np.argmax(predictions)]
        is_poisonous = "Poisonous" if poisonous_indicator[np.argmax(predictions)] == 1 else "Not Poisonous"
        st.write(f"Predicted Mushroom Species: {predicted_species}")
        st.write(f"And it is: {is_poisonous}")
        
        fig,ax = plt.subplots()
        y_pos = np.arange(len(mushroom_classes))
        ax.barh(y_pos, predictions[0], align= 'center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(mushroom_classes)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_title('Mushroom Prediction')

        st.pyplot(fig)
    else:
        st.text('Please upload a valid image.')

if __name__ == '__main__':
    main()
