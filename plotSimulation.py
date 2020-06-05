
import streamlit as st

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import altair as alt



def plotting_SIR_Simulation(S, I, R ,N,t,simulation_period,deaths):

    # Plot the data on three separate curves for S(t), I(t) and R(t)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, 100*S / N, 'b', lw=2, label='Susceptible')

    ax.plot(t, 100*I / N, 'r', lw=2, label='Infected')

    ax.plot(t, 100*R / N, 'g', lw=2, label='Recovered with immunity')

    ax.set_xlabel('Time /days')

    ax.set_ylabel('Percentage of Population (%)')

    ax.set_ylim(0, 100)

    st.write('Total infected cases will be ',int(I[499]))

    st.write('Total recoverd cases will be ',int(R[499]))

    st.write('Total deaths will be ',int(I[499]*deaths))

    #ax.vlines(x=simulation_period,ymin=0,ymax=100,linewidth=2, color='k',label='Intersection')

    legend = ax.legend()

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)

    plt.show()

    st.pyplot()



def plotting_SIR_Infection(S, I, R ,N,t,simulation_period):

    # Plot the data on three separate curves for S(t), I(t) and R(t)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, I, 'r', lw=2, label='Infected')

    ax.set_xlabel('Time /days')

    ax.set_ylabel('Total Infected')

    #ax.set_ylim(0, I)

    st.write('Total infected cases will be ',int(I[499]))

    #ax.vlines(x=simulation_period,ymin=0,ymax=I,linewidth=2, color='k',label='Intersection')

    legend = ax.legend()

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)

    plt.show()

    st.pyplot()





def plotting_SIR_Susceptible(S, I, R ,N,t,simulation_period):

    # Plot the data on three separate curves for S(t), I(t) and R(t)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, S, 'b', lw=2, label='Susceptible')

    ax.set_xlabel('Time /days')

    ax.set_ylabel('Total Susceptible)')

    #ax.set_ylim(0, N)

    st.write('Total Susceptible cases will be ',int(S[499]))

    legend = ax.legend()

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)

    plt.show()

    st.pyplot()



def plotting_SIR_Recovery(S, I, R ,N,t,simulation_period):

    # Plot the data on three separate curves for S(t), I(t) and R(t)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, R, 'g', lw=2, label='Recovery')

    ax.set_xlabel('Time /days')

    ax.set_ylabel('Total Recovery)')

    #ax.set_ylim(0, I)

    st.write('Total Recovered cases will be ',int(R[499]))

    legend = ax.legend()

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)

    plt.show()

    st.pyplot()



def plotting_SIR_IR(S, I, R ,N,t,simulation_period):

    # Plot the data on three separate curves for S(t), I(t) and R(t)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, I, 'r', lw=2, label='Infected')

    ax.plot(t, R, 'g', lw=2, label='Recovery')

    ax.set_xlabel('Time /days')

    ax.set_ylabel('Total Infected)')

    #ax.set_ylim(0, I)

    st.write('Total Infected cases will be ',int(I[499]))

    st.write('Total Recovered cases will be ',int(R[499]))

    legend = ax.legend()

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)

    plt.show()

    st.pyplot()