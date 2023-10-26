image_paths = {
    "setosa": "setosa.jpg",
    "versicolor": "versicolor.jpg",
    "virginica": "virginica.jpg"
}





import streamlit as st
import joblib
import pandas as pd


# Cargar los modelos desde los archivos .sav
svm_model = joblib.load('svm_model.sav')
logistic_model = joblib.load('logistic_model.sav')
tree_model = joblib.load('tree_model.sav')


# Crear una interfaz de usuario con Streamlit
st.title("Actividad Iris Dataset")


# Crear inputs para ingresar los datos
st.sidebar.header("Ingresar datos para la predicción:")
sepal_length = st.sidebar.number_input("Largo del sépalo (cm):")
sepal_width = st.sidebar.number_input("Ancho del sépalo (cm):")
petal_length = st.sidebar.number_input("Largo del pétalo (cm):")
petal_width = st.sidebar.number_input("Ancho del pétalo (cm):")


# Crear un botón para realizar la predicción
if st.sidebar.button("Realizar Predicción"):
    # Crear un DataFrame con los datos ingresados
    data = pd.DataFrame({
        "sepal_length": [sepal_length],
        "sepal_width": [sepal_width],
        "petal_length": [petal_length],
        "petal_width": [petal_width]
    })


    # Realizar predicciones con los modelos
    y_pred_svm = svm_model.predict(data)
    y_pred_logistic = logistic_model.predict(data)
    y_pred_tree = tree_model.predict(data)


    #st.subheader("Resultados de la predicción:")


    #st.write("Predicción con SVM:", y_pred_svm[0])
    #st.write("Predicción con Regresión Logística:", y_pred_logistic[0])
   # st.write("Predicción con Decision Trees:", y_pred_tree[0])




    st.subheader("Resultados de la predicción:")

    st.write("Predicción con SVM:", y_pred_svm[0])
    st.image(image_paths.get(y_pred_svm[0], "imagen_no_encontrada.png"))

    st.write("Predicción con Regresión Logística:", y_pred_logistic[0])
    st.image(image_paths.get(y_pred_logistic[0], "imagen_no_encontrada.png"))

    st.write("Predicción con Decision Trees:", y_pred_tree[0])
    st.image(image_paths.get(y_pred_tree[0], "imagen_no_encontrada.png"))
        




