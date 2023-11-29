import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd

from PIL import Image

import streamlit as st
import pandas as pd
def main():
    # Sample DataFrame
    data = {'col1': [0], 'col2': [0], 'col3': [0]}
    df = pd.DataFrame(data)

    # Lists A and B
    list_A = ['col3', 'col2','col2']
    list_B = ['Option 3', 'Option 1', 'Option 2']  # Replace with your actual choices

    # Streamlit app
    st.title('Select Slider Example')

    # Display the DataFrame
    st.dataframe(df)

    # Collect information from the user using select slider
    selected_option = st.select_slider('Select an option', options=list_B)

    # Get the index of the selected option in list B
    selected_index = list_B.index(selected_option)

    # Get the corresponding column name from list A
    selected_column = list_A[selected_index]

    # Update the corresponding column to 1 in the DataFrame
    df[selected_column] = 1

    # Display the updated DataFrame
    st.title('Updated DataFrame')
    st.dataframe(df)
            

if __name__ == '__main__':
    main()





