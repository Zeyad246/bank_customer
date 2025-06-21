import streamlit as st
import pandas as pd 
import sklearn
import category_encoders
import joblib
import plotly.express as px
st.set_page_config(layout='wide')
df=pd.read_csv('final_churn')
msk = df['Exited'] == 1 
tab1 , tab2 =st.tabs(['why churn' , 'Model'])
with tab1:
    col1 , col2 , col3 , col4 ,col5 ,col6 ,col7,col8,col9,col10 =st.columns(10)
    with col1:
        st.plotly_chart(px.histogram(data_frame=df , x='Card Type',color='Exited' ,title='percent of people whose left the bank', text_auto=True).update_xaxes(categoryorder = 'total descending'))
    with col2:
        st.plotly_chart(px.histogram(data_frame=df , x='CreditScore1',color='Exited', barmode='group' ,title='credit score of every customer', text_auto=True,).update_xaxes(categoryorder = 'total descending'))
    with col3:
        st.plotly_chart(px.histogram(data_frame=df , x='Age' , color='Exited', barmode='group',nbins=40))
    with col4:
        st.plotly_chart(px.histogram(data_frame=df, x='Complain1',color='Exited',histfunc='avg' ,title='percent of complain', barmode='group',text_auto=True).update_xaxes(categoryorder = 'total descending'))
    with col5:
        st.plotly_chart( px.histogram(data_frame=df , x='ActiveStatus',color='Exited',barmode='group',text_auto=True ,title= 'Active status'))
    with col6:
        st.plotly_chart(px.histogram(data_frame=df[msk] , x='Balance',barmode='group',text_auto=True ,title='Balance of people whose left the bank'))
    with col7:
        st.plotly_chart(px.pie(data_frame=df[msk] , names='Geography' , facet_row='Exited'))
    with col8:
        st.plotly_chart(px.histogram(data_frame=df , x='EstimatedSalary',color='Exited' , text_auto=True,title='salary of people').update_xaxes(categoryorder = 'total descending'))
    with col9:
        st.plotly_chart(px.histogram(data_frame=df , x='NumOfProducts',color='Exited',barmode='group',text_auto=True ,title= 'activity'))
    with col10:
        st.plotly_chart(px.histogram(data_frame=df , x='Tenure',color='Exited',barmode='group',text_auto=True ,title= 'How many years has the client been with bank'))
        
with tab2:
    def get_input():
        CreditScore = st.slider('select CreditScore', min_value=350, max_value=850, value=850, step=1)
        Geography = st.selectbox('select Geography', options=['France', 'Germany', 'Spain'])
        Gender = st.selectbox('select gender', options=['Male', 'Female'])
        Age = st.slider('select Age', min_value=18, max_value=92, value=43, step=1)
        Tenure = st.slider('select Tenure', min_value=0, max_value=10, value=2, step=1)
        Balance = st.slider('select Balance', min_value=0, max_value=250898, value=125510, step=1)
        NumOfProducts = st.slider('select  NumOfProducts', min_value=1, max_value=4, value=1, step=1)
        HasCrCard = st.selectbox('select has a card or not [1 for has a card -- 0 for not]', options=[1, 0])
        IsActiveMember=st.selectbox('select active_status [1 for active -- 0 for not]', options=[1, 0])
        EstimatedSalary = st.slider('select Estimated_Salary', min_value=11, max_value=199992, value=79084, step=1)
        Complain= st.selectbox('select has a complain [1 for complain -- 0 for not]', options=[1, 0])
        Satisfaction_Score = st.slider('select Satisfaction_Score', min_value=1, max_value=5, value=5, step=1)
        Card_Type = st.selectbox('select Cardtype', options=['DIAMOND', 'GOLD', 'SILVER', 'PLATINUM'])
        Point_Earned=st.slider('select Estimated_Salary', min_value=119, max_value=1000, value=500, step=1)

        return pd.DataFrame(data=[[CreditScore, Geography, Gender, Age, Tenure, Balance,NumOfProducts,HasCrCard,
                                   IsActiveMember,EstimatedSalary,Complain,Satisfaction_Score,
                                   Card_Type,Point_Earned ]],
                            columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                                     'NumOfProducts','IsActiveMember','HasCrCard', 'EstimatedSalary','Complain','Satisfaction Score',
                                     'Card Type', 'Point Earned',])

    test = get_input()

    if st.button("predict"):
        dt = joblib.load('dt.h5')
        prediction = dt.predict(test)

        st.success(f"Prediction result: {prediction[0]}")

        if prediction[0] == 0:
            st.success("✅ العميل **سيبقى** معنا 🎉")
            st.image("https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif", caption="عميل سعيد", use_column_width=True)
        else:
            st.error("❌ العميل **سيغادر** 😥")
            st.image("https://media.giphy.com/media/3o6Zt481isNVuQI1l6/giphy.gif", caption="حاول تحسين التجربة", use_column_width=True)
