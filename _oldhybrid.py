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
from datetime import datetime
from statistics import mean

def main():

    ## sidebar
    data=load_data('total_data.pkl')
    data1 = data[data['Date'] == data['Date'].iloc[0]]
    countydf=pd.DataFrame(data1.Combined_Key.str.split(',',2).tolist(),columns = ['County','State','Country']).drop('Country',axis=1)
    countydf=pd.DataFrame(countydf.groupby('State')['County'].apply(lambda x: x.values.tolist())).reset_index()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to",

                            ( 'Hybrid Model','Data Exploratory'))


    if page == 'Hybrid Model' :
        '## County Level SIR Simulation Model'
        statesselected = st.selectbox("Select a County", countydf['State'])
        countylist=(countydf[countydf['State']==statesselected]['County']).tolist()[0]
        countyselected = st.selectbox('Select a county for demo',countylist)
        name=countyselected+', '+statesselected.strip()+', '+'US'

        df2=data_cleaning(data,name)
        last_day = data['Date'].max()
        #Maximum prediction range is 14 days
        date_select_box = list((last_day + dt.timedelta(days=x)).strftime('%m-%d-%y') for x in range(2,15))
        selected_pred_day = st.selectbox('Select a date that you will lift some preventative measures', date_select_box)
        train_df = df2[df2['Date'] <  df2.Date.iloc[-7]]
        percentage_influence = np.linspace(10,100,10)
        level = st.selectbox('Estimated percentage to reduce the current preventive measures by',percentage_influence)
        test_df = df2[(df2['Date'] >= df2.Date.iloc[-7]) & (df2['Date'] <= df2.Date.iloc[-1])]
        recent=len(df2)
        Date=df2['Date'][recent-1] 
        dfdate=df2[df2['Date']==Date]
        confirmed = dfdate.loc[dfdate['Date']== Date,'Confirmed'].iloc[0]
        deaths=round(((dfdate.loc[dfdate['Date']== Date,'Deaths'].iloc[0]) / confirmed )*100,3)
        deaths = st.slider("Input a realistic death rate(%) ", 0.0, 30.0, value = deaths)
        simulation_period = st.slider('Input Simulation period (days)',0,100,step = 1,value = 21)
        recovery_day=st.slider('Input recovery period (days)',1,28,step=1,value=14)
        btn = st.button('Run Model')
        
        
        if btn:

            #Model training
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
            I0 = test_df['I'].iloc[-1]
            R0 = test_df['R'].iloc[-1]
            S0 = population - I0 - R0
            est_beta = model.beta
            est_alpha = model.a
            est_b = model.b
            est_c = model.c

            ######################################
            # Add select box for prediction date #
            ######################################
            #last_day = data['Date'].max()
            #Maximum prediction range is 14 days
            #date_select_box = list((last_day + dt.timedelta(days=x)).strftime('%m-%d-%y') for x in range(2,15))
            time_type = datetime.strptime(selected_pred_day, '%m-%d-%y')
            forecast_period = predict_period = (time_type - last_day).days
            st.write('Prediction range from now is ',forecast_period)

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


            #deaths = st.slider("Input a realistic death rate(%) ", 0.0, 30.0, value = deaths)
            result = prediction.run(death_rate=deaths)  # death_rate is an assumption


            #simulation_period = st.slider('Input Simulation period (days)',0,100,step = 1,value = 21)
            #recovery_day=st.slider('Input recovery period (days)',1,28,step=1,value=14)


            #TEST
            ###########################
            #Additional adding by Yang
            ###########################
            betalist=model.show_betalist()
            minbeta=round(min(betalist),4)
            maxbeta=round(max(betalist),4)
            averagebeta=mean(betalist)



            beta=prediction.finalbeta()
            #st.write(beta)
            #userbeta=round((100-(beta*100)), 2)
            #userbeta=st.slider('Input Social distancing factor (%)',0.00,100.00,step = 0.01,value =userbeta)
            #NEW CALCULATION
            maxlimit= (maxbeta * 1.2 - beta * 0.8) / averagebeta
            D= maxlimit / 100
            defaultbeta= (maxbeta * 1.2-beta) / (D * averagebeta)
            defaultbeta_round = round(defaultbeta,4)
            st.write('Current social distancing factor is ',defaultbeta_round)

            #percentage_influence = np.linspace(10,100,10)
            #level = st.selectbox('Estimated percentage to reduce the current preventive measures by',percentage_influence)
            socialdist = defaultbeta_round * (1- level / 100)
            new_beta = round((1.2*maxbeta - socialdist * D * averagebeta),4)
            #st.write(new_beta)

            ###############################################
            ###############################################
            gamma = 1/recovery_day
            social_dist = f'**Social distancing factor of simulation: ** ** {socialdist:2f}**'
            st.markdown(social_dist)
            #st.write('Current Death rate is : ', deaths)

            rr=round(beta/gamma,3)
            R0_current = f'**Effective reproduction number(R0) of current day:** ** {rr:.2f}**'
            st.markdown(R0_current)
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
            I0_simu = result['I'].iloc[-1]
            R0_simu = result['R'].iloc[-1]
            S0_simu = N - I0_simu - R0_simu
            y0_simu = S0_simu, I0_simu, R0_simu

            # Integrate the SIR equations over the time grid, t.

            #ret = odeint(deriv, y0_simu, t, args=(N, new_beta, gamma))
            #S, I, R = ret.T
            #############################
            ##### Add second mitagation date ###
            #############################
            D_t = 500 / simulation_period
            
            ret_first_intervention = odeint(deriv, y0_simu, t, args=(N, new_beta, gamma))

            S, I, R = ret_first_intervention.T
            #mitigation_split_location = round(D_t * mitigation_date)
            #mitigation_y0 = S[mitigation_split_location],I[mitigation_split_location],R[mitigation_split_location]
            #mitigation_t = t[mitigation_split_location:]
            #mitigation_ret = odeint(deriv,mitigation_y0,mitigation_t,args=(N,mitigation_beta,gamma))
            #M_S, M_I, M_R = mitigation_ret.T
            #plot_S = np.concatenate((S[0:(mitigation_split_location)], M_S))
            #plot_I = np.concatenate((I[0:(mitigation_split_location)], M_I))
            #plot_R = np.concatenate((R[0:(mitigation_split_location)], M_R))

            #PLOTTING
            #plotting_SIR_Susceptible(plot_S, plot_I, plot_R ,N, t,simulation_period)

            #plotting_SIR_Infection(plot_S, plot_I, plot_R ,N, t,simulation_period)
            pred_data_date = result['Time'].iloc[0:(forecast_period + 1)] +df2['Day'].max()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df2['Day']-1,df2['I']-1,color = 'b',lw = 2,label = 'Actual data')
            ax.plot(pred_data_date-1, result['I'].iloc[0:(predict_period+2)], color = 'r',label = 'Predictive data')
            ax.plot((t+pred_data_date.iloc[-1]-1),I,color = 'r',linestyle='dashed',label = 'Simulation data')
            legend = ax.legend()
            for spine in ('top', 'right', 'bottom', 'left'):
                ax.spines[spine].set_visible(False)
            plt.show()
            st.pyplot()
            R0_simu = f'**Effective reproduction number(R0) of simulation: ** **{(new_beta / gamma):2f}**'
            st.markdown(R0_simu)
            st.write('Susceptible cases by the end of simulation will be ',int(S[499]))
            st.write('Recovered cases by the end of simulation will be ',int(R[499]))
            I_int = int(I[499])
            Infected_mark = f'**Infected cases by the end of simulation will be ** **{I_int}**'
            st.markdown(Infected_mark)
            D_int =int(deaths *(int(I[499])+int(R[499]))/100)
            Death_mark = f'**Total Death cases by the end of simulation will be ** **{D_int}**'
            st.markdown(Death_mark) 
            #plotting_SIR_Recovery(plot_S, plot_I, plot_R,N, t,simulation_period)

            #####################
            ## Mitigation end ###
            #####################
            #plotting_SIR_Simulation(S, I, R ,N,t,simulation_period,deaths)
            #plotting_SIR_Susceptible(S, I, R ,N, t,simulation_period)
            #plotting_SIR_Infection(S, I, R ,N, t,simulation_period)
            #plotting_SIR_Recovery(S, I, R ,N, t,simulation_period)
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
