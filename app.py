import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

def main():
    ################ Step 1 Create Web Title #####################

    st.title("Binary Classification Streamlit App")
    st.sidebar.title("Binary Classification Streamlit App")
    st.markdown(" เห็ดนี้แซ่บได้หรือไม่??? 🍄‍🟫🍄‍🟫🍄‍🟫")
    st.sidebar.markdown(" เห็ดนี้แซ่บได้หรือไม่??? 🍄‍🟫🍄‍🟫🍄‍🟫")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR = os.path.join(BASE_DIR, 'images')
    image_path = os.path.join(IMAGE_DIR, 'welcome_image.png')
    if 'popup_shown' not in st.session_state:
        st.session_state.popup_shown = False

    if not st.session_state.popup_shown:
        with st.expander("🎉 Welcome! Click to close", expanded=True):
            st.image(image_path, caption="Welcome to Mushroom Classification App!", width=300)
            st.write("This app helps you classify mushrooms as edible or poisonous. Enjoy exploring!")
        st.session_state.popup_shown = True
    
    ############### Step 2 Load dataset and Preprocessing data ##########

    # create read_csv function
    @st.cache_data(persist=True) #ไม่ให้ streamlit โหลดข้อมูลใหม่ทุกๆครั้งเมื่อ run
    def load_data():
        # Define the base directory as the folder containing your script
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        file_path = os.path.join(DATA_DIR, 'mushrooms.csv')

        data = pd.read_csv(file_path)
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache_data(persist=True)
    def spliting_data(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, display_labels=class_names)
            st.pyplot(fig)
    
            
        
        if 'ROC Curve' in metrics_list:
            
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)
      
    df = load_data()
    x_train, x_test, y_train, y_test = spliting_data(df)
    class_names = ['edible','poisonous']
    st.sidebar.subheader("Choose Classifiers")
    classifier  = st.sidebar.selectbox("Classifier", ("Support Vectore Machine (SVM)", "Logistice Regression", "Random Forest"))


     ############### Step 3 Train a SVM Classifier ##########

    if classifier == 'Support Vectore Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma  = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Supper Vector Machine (SVM) results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test, y_test)
            y_pred   = model.predict(x_test)

            precision = precision_score(y_test, y_pred).round(2)
            recall = recall_score(y_test, y_pred).round(2)
            
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", precision)
            st.write("Recall: ", recall)
            plot_metrics(metrics)


    

     ############### Step 4 Training a Logistic Regression Classifier ##########
     # Start you Code here #
    if classifier == 'Logistice Regression':
        st.subheader("Model hyperparameters")
        
        solver_options = {
            "liblinear (Not recommand)" : "liblinear",
            "sag (Stochastic Average Gradient)" : "sag",
            "saga" : "saga",
            "lbfgs (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)" : "lbfgs"
        }
        
        C = st.sidebar.number_input("C (Inverse Regularization Strength)", 0.01 , 10.0 , step=0.01 , key='C' )
        # Show radio options with descriptions (Show keys in dictionary )
        solver_choice = st.sidebar.radio("Solver", list(solver_options.keys()),key = 'solver')
        # Map (use only string (Value in dictionary ) )
        solver = solver_options[solver_choice]
        
        if solver in ['saga','liblinear']:
            penalty = st.sidebar.radio("Penalty",("l1","l2"),key='penalty')
        else:
            penalty = 'l2'
        
        max_iter = st.sidebar.number_input("Max_iter",50,500,step=10,key ='max_iter')
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix","ROC Curve","Precision-Recall Curve"),key='metrics')
        
        if st.sidebar.button("Classify",key = "classify"):
            if solver in ['saga','liblinear']:
                model = LogisticRegression(C=C,solver=solver,max_iter=max_iter,penalty=penalty)
            else:
                model = LogisticRegression(C=C,solver=solver,max_iter=max_iter,penalty=penalty)
            
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)

            precision = precision_score(y_test,y_pred).round(2)
            recall = recall_score(y_test,y_pred).round(2)
            
            st.write("Accuracy: ", round(accuracy,2))
            st.write("Precision: ", round(precision,2))
            st.write("Recall: ", round(recall,2))
            plot_metrics(metrics)

     ############### Step 5 Training a Random Forest Classifier ##########
    # Start you Code here #
    if classifier == 'Random Forest':
        st.subheader("Model hyperparameters")
        
        
        n_estimators = st.sidebar.number_input("n_estimators (100-400)",100,400,step = 20, key ='n_estimators')
        max_depth = st.sidebar.number_input("max_depth (3-20)",3,20,step = 1, key ='max_depth')
        min_samples_split = st.sidebar.number_input("min_samples_split (2-12)",2,12,step = 1, key ='min_samples_split')
        min_samples_leaf = st.sidebar.number_input("min_samples_leaf (1-5)",1,5,step = 1, key ='min_samples_leaf')
        max_features = st.sidebar.radio("max_features",("auto","sqrt","log2"), key ='max_features')
        bootstrap = st.sidebar.radio("bootstrap",(True,False), key ='bootstrap')
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix","ROC Curve","Precision-Recall Curve"),key='metrics')

        if st.sidebar.button("Classify",key = "classify"):
            max_features = None if max_features == "auto" else max_features
            
            model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_features=max_features,bootstrap=bootstrap)
            
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)

            precision = precision_score(y_test,y_pred).round(2)
            recall = recall_score(y_test,y_pred).round(2)
            
            st.write("Accuracy: ", round(accuracy,2))
            st.write("Precision: ", round(precision,2))
            st.write("Recall: ", round(recall,2))
            plot_metrics(metrics)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom dataset")
        st.write(df)

    
    
    




if __name__ == '__main__':
    main()


