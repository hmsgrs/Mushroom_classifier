import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd

from PIL import Image

def load_image_classifier_model():
    with open('..\\models\\cnn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def load_categorical_model():
    with open('..\\models\\model_sup_final.pkl', 'rb') as file:
        model = pickle.load(file)
    return model



def main():
    st.title('Mushroom Classifier')
    st.write('Welcome to your categorical and photographical Mushroom Classifier')
    st.write('Choose the model you want to use:')

    model_options = ['Image Classifier', 'Category Model']
    selected_model = st.selectbox('Select Model:', model_options)

    if selected_model == 'Image Classifier':
        model = load_image_classifier_model()
        st.subheader('You selected the Image classifier.')
        file= st.file_uploader('Please, upload your mushroom photo', type=['jpg','png'])
        if file:
            image = Image.open(file)
            st.image(image, use_column_width=True)

            resized_image = image.resize((48,48))
            img_array = np.array(resized_image) /255
            img_array = img_array.reshape(1,48,48,3)
            

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

    elif selected_model == 'Category Model':
        image = Image.open('Anatomy.jpg')
        lista_columnas = ['cap-diameter', 'stem-height', 'stem-width', 'has-ring', 'cap-shape_b',
 'cap-shape_c', 'cap-shape_f', 'cap-shape_o', 'cap-shape_p', 'cap-shape_s', 'cap-shape_x',
 'cap-surface_d', 'cap-surface_e', 'cap-surface_g', 'cap-surface_h', 'cap-surface_i', 'cap-surface_k', 'cap-surface_l', 'cap-surface_nan',
 'cap-surface_s', 'cap-surface_t', 'cap-surface_w', 'cap-surface_y', 'cap-color_b', 'cap-color_e', 'cap-color_g', 'cap-color_k',
 'cap-color_l', 'cap-color_n', 'cap-color_o', 'cap-color_p', 'cap-color_r', 'cap-color_u', 'cap-color_w', 'cap-color_y',
 'gill-color_b', 'gill-color_e', 'gill-color_f', 'gill-color_g', 'gill-color_k', 'gill-color_n', 'gill-color_o', 'gill-color_p',
 'gill-color_r', 'gill-color_u', 'gill-color_w', 'gill-color_y', 'stem-surface_f', 'stem-surface_g', 'stem-surface_h', 'stem-surface_i',
 'stem-surface_k', 'stem-surface_nan', 'stem-surface_s', 'stem-surface_t', 'stem-surface_y', 'stem-color_b', 'stem-color_e', 'stem-color_f', 'stem-color_g',
 'stem-color_k', 'stem-color_l', 'stem-color_n', 'stem-color_o', 'stem-color_p', 'stem-color_r', 'stem-color_u', 'stem-color_w', 'stem-color_y', 'veil-color_e',
 'veil-color_k', 'veil-color_n', 'veil-color_nan', 'veil-color_u', 'veil-color_w', 'veil-color_y', 'ring-type_e', 'ring-type_f', 'ring-type_g', 'ring-type_l',
 'ring-type_m', 'ring-type_nan', 'ring-type_p', 'ring-type_r', 'ring-type_z', 'habitat_d', 'habitat_g', 'habitat_h', 'habitat_l', 'habitat_m', 'habitat_p', 
 'habitat_u', 'habitat_w', 'season_a', 'season_s', 'season_u', 'season_w']

        model_inputer = pd.DataFrame(data = [[0] * len(lista_columnas)],columns= lista_columnas)
        st.dataframe(model_inputer)

        

        st.image(image, caption='Different parts of a mushroom')
        model = load_categorical_model()
        st.subheader('You selected the Categorical method.')
        st.write('Please use the sliders to select the values of the following categories.')
        st.write()

        st.subheader('Cap diameter')
        cap_diameter_intro = st.slider('in CM, between 1 and 20:',min_value=1.0,max_value=20.0,step=0.5)
        model_inputer['cap-diameter'] = cap_diameter_intro


        st.subheader('Stem height')
        stem_height_intro = st.slider('between 1 and 20:',min_value=1.0,max_value=20.0,step=0.5)
        model_inputer['stem-height'] = stem_height_intro

        st.subheader('Stem width')
        stem_width_intro = st.slider('in CM, between 1 and 10:',min_value=1.0,max_value=10.0,step=0.5)
        model_inputer['stem-width'] = stem_width_intro


        st.subheader('Has ring')
        has_ring_intro = st.slider('0 for no, 1 for yes:',min_value=0,max_value=1,step=1)
        model_inputer['has-ring'] = has_ring_intro

        st.subheader('Stem surface')
        stem_surface_dic= ['stem-surface_g', 'stem-surface_h', 'stem-surface_i', 'stem-surface_k',
   'stem-surface_nan', 'stem-surface_s', 'stem-surface_t', 'stem-surface_y']
        stem_surface= ['grooves', 'shiny', 'fibrous', 'silky',
   'none', 'smooth', 'sticky', 'scaly']
        stem_surface_intro = st.select_slider('Choose category',options=stem_surface)

        selected_index = stem_surface.index(stem_surface_intro)
        selected_column = stem_surface_dic[selected_index]
        model_inputer[selected_column] = 1


        st.subheader('Stem color')
        stem_color_dic= ['stem-color_b', 'stem-color_e', 'stem-color_f', 'stem-color_g', 'stem-color_k', 'stem-color_l',
   'stem-color_n', 'stem-color_o', 'stem-color_p', 'stem-color_r', 'stem-color_u', 'stem-color_w', 'stem-color_y']
        stem_color= ['buff', 'red', 'none', 'gray', 'black', 'blue',
   'brown', 'orange', 'pink', 'green', 'purple', 'white', 'yellow']
        stem_color_intro = st.select_slider('Choose category',options=stem_color)

        selected_index = stem_color.index(stem_color_intro)
        selected_column = stem_color_dic[selected_index]
        model_inputer[selected_column] = 1



        st.subheader('Cap shape')
        cap_shape_dic= ['cap-shape_b', 'cap-shape_c', 'cap-shape_f', 'cap-shape_o', 'cap-shape_p', 'cap-shape_s', 'cap-shape_x']
        cap_shape= ['bell', 'conical', 'flat', 'others', 'spherical', 'sunken', 'convex']
        cap_shape_intro = st.select_slider('Choose category',options=cap_shape)

        selected_index = cap_shape.index(cap_shape_intro)
        selected_column = cap_shape_dic[selected_index]
        model_inputer[selected_column] = 1


        st.subheader('Cap surface')
        cap_surface_dic= ['cap-surface_e', 'cap-surface_g', 'cap-surface_h', 'cap-surface_i', 'cap-surface_k',
 'cap-surface_l', 'cap-surface_nan', 'cap-surface_s', 'cap-surface_t', 'cap-surface_w', 'cap-surface_y']
        cap_surface= ['fleshy', 'grooves', 'shiny', 'fibrous', 'silky',
 'leathery', 'cap-surface_nan', 'smooth', 'sticky', 'wrinkled', 'scaly']
        cap_surface_intro = st.select_slider('Choose category',options=cap_surface)

        selected_index = cap_surface.index(cap_surface_intro)
        selected_column = cap_surface_dic[selected_index]
        model_inputer[selected_column] = 1

        st.subheader('Cap color')
        cap_color_dic= ['cap-color_b', 'cap-color_e', 'cap-color_g', 'cap-color_k', 'cap-color_l', 'cap-color_n',
 'cap-color_o', 'cap-color_p', 'cap-color_r', 'cap-color_u', 'cap-color_w', 'cap-color_y']
        cap_color= ['buff', 'red', 'gray', 'black', 'blue', 'brown',
 'orange', 'pin', 'green', 'purple', 'white', 'yellow']
        cap_color_intro = st.select_slider('Choose category',options=cap_color)

        selected_index = cap_color.index(cap_color_intro)
        selected_column = cap_color_dic[selected_index]
        model_inputer[selected_column] = 1


        st.subheader('Gill color')
        gill_color_dic= ['gill-color_b', 'gill-color_e', 'gill-color_f', 'gill-color_g', 'gill-color_k', 'gill-color_n', 
  'gill-color_o', 'gill-color_p', 'gill-color_r', 'gill-color_u', 'gill-color_w', 'gill-color_y']
        gill_color= ['buff', 'red', 'none', 'gray', 'black', 'brown', 'orange', 'pink', 'green', 'purple', 'white', 'yellow']        
        gill_color_intro = st.select_slider('Choose category',options=gill_color)

        selected_index = gill_color.index(gill_color_intro)
        selected_column = gill_color_dic[selected_index]
        model_inputer[selected_column] = 1


        st.subheader('Veil color')
        veil_color_dic= ['veil-color_e', 'veil-color_k', 'veil-color_n', 'veil-color_nan', 'veil-color_u', 'veil-color_w', 'veil-color_y']
        veil_color= ['red','black','brown','no color','purple','white','yellow']
        veil_color_intro = st.select_slider('Choose category',options=veil_color)

        selected_index = veil_color.index(veil_color_intro)
        selected_column = veil_color_dic[selected_index]
        model_inputer[selected_column] = 1

        
        st.subheader('Ring type')
        ring_type_dic= [ 'ring-type_e', 'ring-type_r', 'ring-type_l', 'ring-type_p', 'ring-type_z']
        ring_type= ['evanescent','flaring','grooved','large','pendant','zone']
        ring_type_intro = st.select_slider('Choose category',options=ring_type)


        selected_index = ring_type.index(ring_type_intro)
        selected_column = ring_type_dic[selected_index]
        model_inputer[selected_column] = 1



        st.subheader('habitat')
        habitat = ['grasses','leaves','meadow','paths', 'heaths', 'urban', 'waste,', 'woods']
        habitat_dic= ['habitat_g', 'habitat_l', 'habitat_m', 'habitat_p', 'habitat_h', 'habitat_u', 'habitat_w', 'habitat_d']
        habitat_intro = st.select_slider('Choose category',options=habitat)

        selected_index = habitat.index(habitat_intro)
        selected_column = habitat_dic[selected_index]
        model_inputer[selected_column] = 1




        st.subheader('season')
        season= ['autumn','spring','summer','winter']
        season_dic= ['season_a', 'season_s', 'season_u', 'season_w']   
        season_intro = st.select_slider('Choose category',options=season)

        selected_index = season.index(season_intro)
        selected_column = season_dic[selected_index]
        model_inputer[selected_column] = 1
        
        
        with open('..\\data\\processed\\poison_dict.pkl', 'rb') as file:
            poison_dict = pickle.load(file)
        substitution_mapping = {'e': 'edible', 'p': 'poisonous'}


        poison_dict = {key: substitution_mapping[value] for key, value in poison_dict.items()}
        
        st.dataframe(model_inputer)
        predictions= model.predict(model_inputer)

        
        key = predictions[0]
        value = poison_dict[key]
        st.subheader('That would be a ' + predictions + ' and it is ' + value)
        
        
       
        
        

if __name__ == '__main__':
    main()





