import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
#import seaborn as sns; sns.set()
from helpers import *
from SIR_Model import *
from scipy.integrate import odeint
import matplotlib.animation as animation
import datetime as dt
from plotSimulation import *
from datacleaning import *
import datetime as dt
from statistics import mean

def main():
    ## sidebar
    data=load_data('total_data.pkl')
    countydf=pd.DataFrame(data.Combined_Key.str.split(',',2).tolist(),columns = ['County','State','Country']).drop('Country',axis=1)
    countydf=pd.DataFrame(countydf.groupby('State')['County'].apply(lambda x: x.values.tolist())).reset_index()


    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to",
                            ( 'Hybrid Model','Data Exploratory'))

    if page == 'Hybrid Model' :
        '## County Level SIR Simulation Model'
        statesselected = st.selectbox("Select a State", countydf['State'])
        countylist=(countydf[countydf['State']==statesselected]['County']).tolist()[0]
        countyselected = st.selectbox('Select a County',countylist)

        name=countyselected+', '+statesselected.strip()+', '+'US'

        df2=data_cleaning(data,name)

        #data=data[data['Combined_Key']==name]
        #df2 = load_data('{}.pkl'.format(selected.lower()))

        #Model training
        train_df = df2[df2['Date'] < df2.Date.iloc[-7]]
        test_df = df2[(df2['Date'] > df2.Date.iloc[-7]) & (df2['Date'] < df2.Date.iloc[-1])]

        # initialize model
        #'## Training the Model'
        with st.spinner('Model Training in Progress...'):
            population = df2.Population[1]
            model = Train_Dynamic_SIR(epoch=5000, data=train_df,
                                      population=population, gamma=1 / 15, c=1, b=-10, a=0.08)

            # train the model
            estimate_df = model.train()

        "## Future Forecast"
        # initialize parameters for prediction
        population = model.population
        I0 = train_df['I'].iloc[-1]
        R0 = train_df['R'].iloc[-1]
        S0 = population - I0 - R0
        est_beta = model.beta
        est_alpha = model.a
        est_b = model.b
        est_c = model.c

        forecast_period = 21
        #forecast_period = st.slider("Choose the forecast period(days)", 5, 60,step =5, value=21)

        prediction = Predict_SIR(pred_period=forecast_period, S=S0, I=I0, R=R0, gamma=1 / 14,
                                 a=est_alpha, c=est_c, b=est_b, past_days=train_df['Day'].max())
        recent=len(df2)
        Date=df2['Date'][recent-1]
        dfdate=df2[df2['Date']==Date]

        #Calculating death rate
        N = dfdate.loc[dfdate['Date']== Date,'Population'].iloc[0]
        confirmed = dfdate.loc[dfdate['Date']== Date,'Confirmed'].iloc[0]
        deaths=round(((dfdate.loc[dfdate['Date']== Date,'Deaths'].iloc[0]) / confirmed )*100,3)
        I0 = dfdate.loc[dfdate['Date']== Date,'I'].iloc[0]
        R0 = dfdate.loc[dfdate['Date']== Date,'R'].iloc[0] + dfdate.loc[dfdate['Date']== Date,'Deaths'].iloc[0]

        deaths = st.slider("Input a realistic death rate(%) ", 0.0, 30.0, value = deaths)
        result = prediction.run(death_rate=deaths)  # death_rate is an assumption

        simulation_period = st.slider('Input Simulation period (days)',0,100,step = 1,value = 21)
        recovery_day=st.slider('Input recovery period (%)',1,28,step=1,value=14)

        beta=prediction.finalbeta()
        newbeta=(100-(beta*100))
        userbeta=round(newbeta, 2)
        #st.write(userbeta)
        #userbeta=st.slider('Input Social distancing factor (%)',0.00,100.00,step = 0.01,value =userbeta)

        #Additional adding by Yang
        betalist=model.show_betalist()
        minbeta=round(min(betalist),2)
        maxbeta=round(max(betalist),2)
        averagebeta=mean(betalist)
        beta=prediction.finalbeta()
        userbeta=round((100-(beta*100)), 2)
        #userbeta=st.slider('Input Social distancing factor (%)',0.00,100.00,step = 0.01,value =userbeta)
        #NEW CALCULATION
        maxlimit= round(((maxbeta * 1.1 - beta * 0.9) / averagebeta),2)
        D= maxlimit / 100
        defaultbeta= (maxbeta * 1.1-beta) / (D * averagebeta)
        defaultbeta_round = round(defaultbeta,2)
        socialdist=st.slider('New change Social distancing (%)',0.00,100.00,step = 0.01,value =defaultbeta_round)
        new_beta = round((1.1*maxbeta - socialdist * D * averagebeta),2)

        gamma = 1/recovery_day

        beta=(100-new_beta)/100
        st.subheader('SIR simulation for chosen Date '.format(df2['Date'].dt.date[recent-1]))
        st.write(dfdate[['Date','Population','Confirmed','Recovered','Deaths','Active']])

        st.write('Curent value of (Beta) Social distancing factor : ', beta)
        st.write('Current Death rate is : ', deaths)

        #rr=round(beta/gamma,3)
        rr=round(beta/gamma,2)
        st.write('Effective reproduction number(R0): ', rr)

        S0 = N - I0 - R0
        t = np.linspace(0, simulation_period, 500)

        # The SIR model differential equations.
        def deriv(y, t, N, beta, gamma):
            S, I, R = y
            dSdt = -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            return dSdt, dIdt, dRdt

        # Initial conditions vector
        y0 = S0, I0, R0
        # Integrate the SIR equations over the time grid, t.
        ret = odeint(deriv, y0, t, args=(N, beta, gamma))
        S, I, R = ret.T

        #plotting_SIR_Simulation(S, I, R ,N,t,simulation_period,deaths)
        plotting_SIR_Susceptible(S, I, R ,N, t,simulation_period)
        plotting_SIR_Infection(S, I, R ,N, t,simulation_period)
        plotting_SIR_Recovery(S, I, R ,N, t,simulation_period)
        #plotting_SIR_IR(S, I, R ,N,t,simulation_period)

    else:
        st.title('Explore County Level Data ')
        # load data
        statesselected = st.selectbox("Select a County", countydf['State'])
        countylist=(countydf[countydf['State']==statesselected]['County']).tolist()[0]
        countyselected = st.selectbox('Select a county for demo',countylist)
        name=countyselected+', '+statesselected.strip()+', '+'US'

        df=data_cleaning(data,name)

        # drawing
        base = alt.Chart(df).mark_bar().encode( x='monthdate(Date):O',).properties(width=500)

        red = alt.value('#f54242')
        a = base.encode(y='Confirmed').properties(title='Total Confirmed')
        st.altair_chart(a,use_container_width=True)

        b = base.encode(y='Deaths', color=red).properties(title='Total Deaths')
        st.altair_chart(b,use_container_width=True)

        c = base.encode(y='New Cases').properties(title='Daily New Cases')
        st.altair_chart(c,use_container_width=True)

        d = base.encode(y='New deaths', color=red).properties(title='Daily New Deaths')
        st.altair_chart(d,use_container_width=True)


        dates=df['Date'].dt.date.unique()

        selected_date = st.selectbox('Select a Date to Start',(dates))
        forecastdf=df[df['Date'].dt.date >=selected_date]

        if st.checkbox('Show Raw Data'):
            st.write(forecastdf)

        if st.checkbox('Visualization Chart'):
            df_temp = forecastdf.rename(columns = {'I':'Active Infection Cases','R':'Recovered Cases'})
            e = pd.melt(frame = df_temp,
                        id_vars='Date',
                        value_vars=['Active Infection Cases','Recovered Cases'],
                        var_name = 'type',
                        value_name = 'count')

            e = alt.Chart(e).mark_area().encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('count:Q',title = 'Number of Cases'),
                color = alt.Color('type:O',legend = alt.Legend(title = None,orient = 'top-left'))
            ).configure_axis(
                grid=False
            )

            st.altair_chart(e, use_container_width=True)


    st.title("About")
    st.info(
            "This app uses JHU data available in [Github]"
            "(https://github.com/CSSEGISandData/COVID-19) repository.\n\n")

main()

