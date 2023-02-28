import numpy as np
import pickle
import streamlit as st
import sklearn

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def breast_cancer_prediction(input_data):

    inp_data = np.asarray(input_data)

    inp_datas = inp_data.reshape(1,-1)

    prediction = loaded_model.predict(inp_datas)

    if (prediction==0):
        return 'Benign'
    else:
        return 'Malignant'


def main():
    st.title('Breast Cancer Prediction')
    left, right = st.columns([1,1])
    with left:
        radius_mean = st.text_input('Radius Mean')
        texture_mean = st.text_input('Texture Mean')
        perimeter_mean = st.text_input('Perimeter mean')
        area_mean = st.text_input('Area Mean')
        smoothness_mean = st.text_input('Smoothness Mean')
        compactness_mean =  st.text_input('Compactness Mean')
        concavity_mean =  st.text_input('Concavity Mean')
        concave_points_mean = st.text_input('Concave Points Mean')
        symmetry_mean = st.text_input('Symmetry Mean')
        fractal_dimension_mean = st.text_input('Fractal Dimension Mean')
        radius_se = st.text_input('Radius SE')
        texture_se = st.text_input('Texture SE')
        perimeter_se = st.text_input('Perimeter SE')
        area_se = st.text_input('Area SE')
        smoothness_se = st.text_input('Smoothness SE')
    with right:
        compactness_se = st.text_input('Compactness SE')
        concavity_se = st.text_input('Concavity SE')
        concave_points_se = st.text_input('Concave Points SE')
        symmetry_se = st.text_input('Symmetry SE')
        fractal_dimension_se = st.text_input('Fractal Dimension SE')
        radius_worst = st.text_input('Radius Worst')
        texture_worst = st.text_input('Texture Worst')
        perimeter_worst = st.text_input('Perimeter Worst')
        area_worst = st.text_input('Area Worst')
        smoothness_worst = st.text_input('Smoothness Worst')
        compactness_worst = st.text_input('Compactness Worst')
        concavity_worst = st.text_input('Concavity Worst')
        concave_points_worst = st.text_input('Concave Points Worst')
        symmetry_worst = st.text_input('Symmetry Worst')
        fractal_dimension_worst = st.text_input('Fractal Dimension Worst')

    diagnosis = ''

    if st.button('Check Result'):
        diagnosis = breast_cancer_prediction([radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean,
                                              concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
                                              smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, 
                                              texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concavity_worst,
                                              concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
                                              ])
        
    st.success(diagnosis)
    

if __name__ == '__main__':
    main()


