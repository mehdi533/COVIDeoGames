import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.signal import savgol_filter
import json
from datetime import datetime
import statsmodels.api as sm
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
import os
from helper_functions import *

import requests
import time
from causalimpact import CausalImpact

from tqdm.notebook import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import threading
import csv
import gc
from lxml import etree
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from ipywidgets import interact, VBox
from IPython.display import display, clear_output
import plotly.io as pio
from matplotlib.lines import Line2D

import dash
from dash import dcc, html

from scipy.stats import pearsonr

import plotly.express as px
import os
import requests
import time
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Libraries for the plots:
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets
from IPython.display import display, clear_output

# Libraries for the LSTM-based neural network model:
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Libraries for statistical tests:
from causalimpact import CausalImpact
from scipy import stats
from scipy.stats import t

# Library for the wikipedia API
import wikipediaapi

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from causalimpact import CausalImpact
from helper import *


def load_interventions():
    
    interventions = pd.read_csv('interventions.csv')
    interventions.set_index('lang', inplace=True)
    return interventions

def load_applemob():
    applemob = pd.read_csv('applemobilitytrends-2020-04-20.csv')
    return applemob

def load_globalmob():
    globalmob = pd.read_csv('Global_Mobility_Report.csv')
    return globalmob

def load_aggregated_timeseries():
    with open('aggregated_timeseries.json', 'r') as f:
        # Load the JSON data
        d = json.load(f)
    return d

def choose_restrictiveness(choice, english):
    if choice == "All":
        data = {
            'France': ['fr', 'FR'],
            'Denmark': ['da', 'DK'],
            'Germany': ['de', 'DE'],
            'Italy': ['it', 'IT'],
            'Netherlands': ['nl', 'NL'],
            'Norway': ['no', 'NO'],
            'Serbia': ['sr', 'RS'],
            'Sweden': ['sv', 'SE'],
            'Korea': ['ko', 'KR'],
            'Catalonia': ['ca', 'ES'],
            'Finland': ['fi', 'FI'],
            'Japan': ['ja', 'JP'],
            }
    if choice == "Restrictive":
        data = {
            'France': ['fr', 'FR'],
            'Italy': ['it', 'IT'],
            'Serbia': ['sr', 'RS'],
            'Catalonia': ['ca', 'ES'],
            }
    if choice == "Semi-Restrictive":
        data = {
            'Denmark': ['da', 'DK'],
            'Germany': ['de', 'DE'],
            'Netherlands': ['nl', 'NL'],
            'Norway': ['no', 'NO'],
            'Finland': ['fi', 'FI'],
            }
    if choice == "Unrestrictive":
        data = {
            'Sweden': ['sv', 'SE'],
            'Korea': ['ko', 'KR'],
            'Japan': ['ja', 'JP'],
            }
    if english == "Yes":
        data['England'] = ['en', 'GB']

    df_code = pd.DataFrame(data)
    df_code = df_code.transpose()
    df_code.rename(columns = {0:'lang', 1:'state'}, inplace = True)
    return data, df_code

def average_mobility(d, df_code, interventions, globalmob):
    # Create subplot
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Mobility'], vertical_spacing=0.1)

    # Initialize a list to store individual mobility lines
    all_lines = []

    for i, c in enumerate(df_code['lang']):
        cs = df_code.iloc[i]['state']

        if cs == 'KR':
            globalmob_ko = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['sub_region_1'].isnull()) & (globalmob['metro_area'].isnull())]
        else:
            if cs == 'RS':
                globalmob_sr = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['metro_area'].isnull())]
            else:
                if cs == 'ES':
                    globalmob_ca = globalmob[(globalmob['country_region_code'] == 'ES') & (globalmob['sub_region_1'] == 'Catalonia') & (globalmob['sub_region_2'].isnull())].copy()
                else:
                    globalmob_g = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['sub_region_1'].isnull())].copy()
                    globalmob_g.reset_index(inplace=True, drop=True)

        df = globalmob_g.copy(deep=True)

        mobility_g = interventions.loc[c]['Mobility']
        lockdown_g = interventions.loc[c]['Lockdown']
        normalcy_g = interventions.loc[c]['Normalcy']

        columns = globalmob.columns[8:]
        df = df.drop(['residential_percent_change_from_baseline', 'parks_percent_change_from_baseline'], axis=1)
        columns = columns.drop(['residential_percent_change_from_baseline', 'parks_percent_change_from_baseline'])

        mean_g = df[columns].mean(axis=1)

        # Light grey lines for individual countries
        for column in columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df[column], mode='lines', line=dict(color='lightgrey', width=1.5), showlegend=False, opacity=0.25))
        
        # Vertical lines
        if cs =='GB':
            fig.add_trace(go.Scatter(x=[lockdown_g, lockdown_g], y=[-100, 100], mode='lines', line=dict(color='black', width=1.5), name=f'Lockdown', showlegend=True))
            fig.add_trace(go.Scatter(x=[normalcy_g, normalcy_g], y=[-100, 100], mode='lines', line=dict(color='black', width=1.5, dash='dash'), name=f'Normalcy', showlegend=True))
        else:
            fig.add_trace(go.Scatter(x=[lockdown_g, lockdown_g], y=[-100, 100], mode='lines', line=dict(color='black', width=1.5), name=f'Lockdown {c}', showlegend=False))
            fig.add_trace(go.Scatter(x=[normalcy_g, normalcy_g], y=[-100, 100], mode='lines', line=dict(color='black', width=1.5, dash='dash'), name=f'Normalcy {c}', showlegend=False))

        # Plot individual lines
        if cs == 'GB':
            line_label = f'Average Mobility in {df_code.index[i]}'
            line_color = 'red'
            line_width = 4
            fig.add_trace(go.Scatter(x=df['date'], y=mean_g, mode='lines', name=line_label, line=dict(color=line_color, width=line_width), showlegend=True))
        else:
            line_label = '_nolegend_'
            line_color = 'grey'
            line_width = 1.5
            fig.add_trace(go.Scatter(x=df['date'], y=mean_g, mode='lines', name=line_label, line=dict(color=line_color, width=line_width), showlegend=False, opacity=0.5))

        # Add individual lines to the list
        all_lines.append(mean_g)

    # Calculate the average line for all countries
    average_line = np.mean(all_lines, axis=0)

    # Plot the average line as a thick blue line
    fig.add_trace(go.Scatter(x=df['date'], y=average_line, mode='lines', name='Average Mobility (All Countries)', line=dict(color='blue', width=4)))

    # Customize layout
    fig.update_layout(
        title='Comparing Normalized Percentage of Wikipedia page views related to video games to English',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Percentage of Mobility Compared to Day 0'),
        legend=dict(x=1.02, y=1),
        showlegend=True,
        height=600,
        width=900,
    )

    fig.show()

    # Output html that you can copy paste
    #fig.to_html(full_html=False, include_plotlyjs='cdn')
    # Saves a html doc that you can copy paste
    #fig.write_html("average_mobility.html", full_html=False, include_plotlyjs='cdn')
    return

def plot_percent_pageviews(d, df_code, interventions):
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Percentage of Wikipedia page views related to video games'])
    all_lines = []
    max_length = 0  # Track the maximum length of y_fit arrays

    for i, c in enumerate(df_code['lang']):
        dt = d[c]["topics"]["Culture.Media.Video games"]["percent"]

        mobility_g = interventions.loc[c]['Mobility']
        format_string = "%Y-%m-%d"

        # Convert the string to a numpy.datetime64 object
        date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

        dates = list(dt.keys())
        numbers = list(dt.values())

        dates = pd.to_datetime(dates)

        if c == 'sv':
            x = [datetime.timestamp(k) for k in dates]
            x = x[365:]
            y = [val for val in numbers if not math.isnan(val)]
        else:
            x = [datetime.timestamp(k) for k in dates]
            y = numbers

        degree = 4
        coefficients = np.polyfit(x, y, degree)
        polynomial = np.poly1d(coefficients)

        y_fit = polynomial(x)

        # Track the maximum length
        max_length = max(max_length, len(y_fit))

        # Plot individual lines
        fig.add_trace(go.Scatter(x=dates, y=numbers, mode='lines', line=dict(color='lightgrey', width=0.5), showlegend=False, opacity=0.5))
        
        if c =='fr':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='green', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='da':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='orange', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='de':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='yellow', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='it':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='purple', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='nl':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='pink', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='no':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='black', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='sr':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='grey', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='sv':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='brown', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='ko':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='mediumorchid', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='ca':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='tan', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='fi':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='olive', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='ja':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='dodgerblue', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='en':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='lawngreen', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))

        fig.add_shape(go.layout.Shape(
            type='line',
            x0=date_object,
            x1=date_object,
            y0=0,
            y1=1,
            line=dict(color='blue', width=1.5, dash='dash'),
            layer='below'
        ))

        # Add individual lines to the list
        all_lines.append(y_fit)

    # Pad shorter arrays with NaN values
    all_lines_padded = [np.pad(line, (0, max_length - len(line)), 'constant', constant_values=np.nan) for line in all_lines]

    # Calculate the average line for all countries
    average_line = np.nanmean(all_lines_padded, axis=0)

    # Plot the average line as a thick blue line
    fig.add_trace(go.Scatter(x=dates, y=average_line, mode='lines', name='Average Trend', line=dict(color='red', width=3)))

    # Update layout
    fig.update_layout(
        xaxis=dict(title='Date', tickangle=45, tickmode='array'),
        yaxis=dict(title='Percentage', range=[0, 0.015]),
        showlegend=True,
        height=600,
        width=800,
    )

    fig.show()
    # Output html that you can copy paste
    #fig.to_html(full_html=False, include_plotlyjs='cdn')
    # Saves a html doc that you can copy paste
    #fig.write_html("percent_pageviews.html", full_html=False, include_plotlyjs='cdn')
    return

def plot_normalized_percent_pageviews(d, df_code, interventions):
# Assuming df_code, d, and interventions are defined as in your Matplotlib code

    fig = go.Figure()

    all_lines = []
    max_length = 0  # Track the maximum length of y_fit arrays

    for i, c in enumerate(df_code['lang']):
        dt = d[c]["topics"]["Culture.Media.Video games"]["percent"]

        mobility_g = interventions.loc[c]['Mobility']
        format_string = "%Y-%m-%d"

        # Convert the string to a numpy.datetime64 object
        date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

        dates = list(dt.keys())
        numbers = list(dt.values())

        dates = pd.to_datetime(dates)

        if c == 'en':
            x = [datetime.timestamp(k) for k in dates]
            y = [val for val in numbers if not math.isnan(val)]
            line_color = 'red'
            line_width = 6  # Set thickness for England line
            x2 = [datetime.timestamp(k) for k in dates]
            y2 = [val for val in numbers if not math.isnan(val)]

        else:
            x = [datetime.timestamp(k) for k in dates]
            y = numbers
            line_color = 'grey'
            line_width = 2  # Set default thickness for other lines

        degree = 4
        coefficients = np.polyfit(x, y, degree)
        polynomial = np.poly1d(coefficients)

        y_fit = polynomial(x)

        index = dates.get_loc(date_object)
        mean = y_fit[0:index].mean()
        offset = 0 - mean
        y_fit = y_fit + offset

        if c == 'en':
            fig.add_trace(go.Scatter(x=pd.to_datetime(x, unit='s'), y=y_fit,
                                mode='lines',
                                name=f'{df_code.index[i]} - Trend Line',
                                line=dict(color=line_color, width=line_width),
                                showlegend=True))
            coefficients2 = np.polyfit(x2, y2, degree)
            polynomial2 = np.poly1d(coefficients2)
            index2 = dates.get_loc(date_object)
            mean2 = y_fit[0:index].mean()
            offset2 = 0 - mean2
            y_fit2 = polynomial(x2)
            y_fit2 = y_fit2 + offset
            y_fit2 = y_fit2 / 2.5
            max_length = max(max_length, len(y_fit2))
            fig.add_trace(go.Scatter(x=pd.to_datetime(x, unit='s'), y=y_fit2,
                                    mode='lines',
                                    name=f'Scaled England - Trend Line',
                                    line=dict(color='orange', width=line_width)))

        max_length = max(max_length, len(y_fit))
        
        if c != 'en':
            fig.add_trace(go.Scatter(x=pd.to_datetime(x, unit='s'), y=y_fit,
                                    mode='lines',
                                    name=f'{df_code.index[i]} - Trend Line',
                                    line=dict(color=line_color, width=line_width),
                                    showlegend=False))

        all_lines.append(y_fit)

    # Pad shorter arrays with NaN values
    all_lines_padded = all_lines
    all_lines_padded = np.array(all_lines_padded)
    # Calculate the average line for all countries
    average_line = np.nanmean(all_lines_padded, axis=0)
    average_line_restrictive = np.nanmean(all_lines_padded[[0,3,6,9,12],:], axis=0)
    average_line_semi = np.nanmean(all_lines_padded[[1,2,4,5,10,12],:], axis=0)
    average_line_unrestrictive = np.nanmean(all_lines_padded[[7,8,11,12],:], axis=0)

    # Plot the average line as a thick blue line
    fig.add_trace(go.Scatter(x=pd.to_datetime(x, unit='s'), y=average_line,
                            mode='lines',
                            name='Average Trend',
                            line=dict(color='blue', width=6)))
    fig.add_trace(go.Scatter(x=pd.to_datetime(x, unit='s'), y=average_line_restrictive,
                            mode='lines',
                            name='Average Restrictive Trend',
                            line=dict(color='blue', width=6)))
    fig.add_trace(go.Scatter(x=pd.to_datetime(x, unit='s'), y=average_line_semi,
                            mode='lines',
                            name='Average Semi-Restrictive Trend',
                            line=dict(color='blue', width=6)))
    fig.add_trace(go.Scatter(x=pd.to_datetime(x, unit='s'), y=average_line_unrestrictive,
                            mode='lines',
                            name='Average Unrestrictive Trend',
                            line=dict(color='blue', width=6),))

    # Add buttons for toggling between different graphs
    buttons = [
        dict(label='All Countries',
            method='update',
            args=[{'visible': [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,False]}],
            name='All Countries'),
        dict(label='Restrictive Countries',
            method='update',
            args=[{'visible': [True,False,False,True,False,False,True,False,False,True,False,False,True,True,False,True,False,False]}]),
        dict(label='Semi-Restrictive Countries',
            method='update',
            args=[{'visible': [False,True,True,False,True,True,False,False,False,False,True,False,True,True,False,False,True,False]}]),
        dict(label='Unrestrictive Countries',
            method='update',
            args=[{'visible': [False,False,False,False,False,False,False,True,True,False,False,True,True,True,False,False,False,True]}]),
    ]

    fig.update_layout(
        title='Comparing Normalized Percentage of Wikipedia page views related to video games to English',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Normalized Percentage'),
        xaxis_range=[min(dates), max(dates)],
        xaxis_tickvals=pd.to_datetime(pd.date_range(start=dates[0], end=dates[-1], freq='90D')),
        xaxis_ticktext=pd.date_range(start=dates[0], end=dates[-1], freq='90D').strftime('%Y-%m-%d'),
        legend=dict(x=0, y=1),
        updatemenus=[{'type': 'buttons',
                    'showactive': True,
                    'buttons': buttons,
                    'active': 0,
                    'x': 1.3,
                    'y': 1}]
    )
    fig.update_traces(visible=False, selector=dict(name='Average Restrictive Trend'))
    fig.update_traces(visible=False, selector=dict(name='Average Semi-Restrictive Trend'))
    fig.update_traces(visible=False, selector=dict(name='Average Unrestrictive Trend'))

    fig.show()

    # Output html that you can copy paste
    #fig.to_html(full_html=False, include_plotlyjs='cdn')
    # Saves a html doc that you can copy paste
    #fig.write_html("normalized_percent.html", full_html=False, include_plotlyjs='cdn')
    return

def plot_mobility(d, df_code, interventions):
    # Sample data
    # Assuming df_code, d, and interventions are defined

    # Create subplot
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Mobility'], vertical_spacing=0.1)

    # Define dropdown menu
    lockdown_types = ['Restrictive', 'Semi-Restrictive', 'Unrestrictive']
    lockdown_countries = {
        'Restrictive': ['fr', 'ca', 'it', 'sr'],
        'Semi-Restrictive': ['da', 'de', 'nl', 'no', 'fi', 'en'],
        'Unrestrictive': ['ko', 'ja', 'sv']
    }

    buttons = [dict(label=lockdown_type, method='update',
                    args=[{'visible': [c in lockdown_countries[lockdown_type] for c in df_code['lang']]}])
            for lockdown_type in lockdown_types]

    fig.update_layout(
        updatemenus=[dict(type='dropdown', active=0, buttons=buttons, x=0.1, y=1.15)],
    )

    for i, c in enumerate(df_code['lang']):
        dt = d[c]["topics"]["Culture.Media.Video games"]["percent"]
        dates = list(dt.keys())
        numbers = list(dt.values())
        dates = pd.to_datetime(dates)

        if c == 'sv':
            x = [datetime.timestamp(k) for k in dates]
            x = x[365:]
            y = [val for val in numbers if not np.isnan(val)]
        else:
            x = [datetime.timestamp(k) for k in dates]
            y = numbers

        # Creating the approximated curve
        degree = 5
        coefficients = np.polyfit(x, y, degree)
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x)

        # Converting the mobility date (str) into a np.datetime64 to be able to use it
        mobility_g = interventions.loc[c]['Mobility']
        format_string = "%Y-%m-%d"
        date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

        # Offset each curve
        index = dates.get_loc(date_object)
        mean = y_fit[0:index].mean()
        offset = 0 - mean
        y_fit = y_fit + offset

        # Convert x back to datetime for plotting
        x_datetime = pd.to_datetime(x, unit='s')

        # Plot the trace
        fig.add_trace(go.Scatter(x=x_datetime, y=y_fit, mode='lines', name=c, visible=False))

    # Show the initial traces
    fig.data[0].visible = True
    fig.data[3].visible = True
    fig.data[6].visible = True
    fig.data[9].visible = True

    # Customize layout
    fig.update_layout(
        height=400,
        width=800,
        showlegend=True,
        legend=dict(x=1.02, y=1.0),  # Set x position greater than 1 to move legend to the right
        xaxis=dict(title='Date', tickangle=45, tickmode='array'),
        yaxis=dict(title='Percentage'),
    )

    fig.show()
    # Output html that you can copy paste
    #fig.to_html(full_html=False, include_plotlyjs='cdn')
    # Saves a html doc that you can copy paste
    #fig.write_html("mobility.html", full_html=False, include_plotlyjs='cdn')
    return

def return_game_figure(column_name, df, interventions, title='views'):

    fig, ax = plt.subplots(figsize=(15, 4))

    # Allows the user to enter 'Call of Duty' instead of 'Call_of_Duty'
    filtered_df = df[[column_name.replace(' ', '_')]]

    # Choose a color palette from seaborn
    color_palette = sns.color_palette("colorblind", len(filtered_df.columns))

    # Convert the color palette to a list
    list_of_colors = color_palette.as_hex()

    # Plotting each subcolumn corresponding to the language
    for column, color in zip(filtered_df.columns, list_of_colors):

        # Plots the number of views for each language with respect to time
        df_lang = interventions[interventions['lang']==column[1]]
        ax.plot(filtered_df.index, filtered_df[column], label=column[1], color=color)

        # Plots the period of lockdown in bold
        start_index = df_lang['Mobility'].item()
        end_index = df_lang['Normalcy'].item()
        limits_df = filtered_df.loc[start_index:end_index]
        ax.plot(limits_df.index, limits_df[column], color=color, linewidth=5)

    # Adding legend, labels, and title
    ax.set_yscale('log')
    ax.legend(loc='upper right',fontsize=7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Views on the page (log scale)')
    if title == 'views':
        ax.set_title(f'Page Views Over Time for {column_name}')
    if title == 'percentage':
        ax.set_title(f'Percentage of Total Wikipedia Views for {column_name}')
    return fig

def return_specific_game():
    interventions = pd.read_csv('interventions.csv')
    country_code = ['en', 'fr', 'de', 'nl', 'fi', 'ja']

    start_dt = '2019100100' #Start day of the search
    end_dt = '2020123100' #End day of the search

    headers = {'User-Agent':'ADABot/0.0 (floydchow7@gmail.com)'}

    # Retrieve page views for the entire wikipedia for a particular country:

    df_wikiviews = pd.DataFrame()

    for country in country_code:
        # Declare f-string for all the different requests:
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/{country}.wikipedia/all-access/user/daily/{start_dt}/{end_dt}"
        try:
            r = requests.get(url, headers=headers)
            df_onequery = pd.DataFrame(r.json()['items'])
            df_wikiviews = pd.concat([df_wikiviews,df_onequery])
            time.sleep(0.5) # In case the IP address is blocked
        except:
            print('The {} page views are not found during these time'.format(country))

    # Drop useless columns and reset index
    df_wikiviews = df_wikiviews[['project', 'timestamp', 'views']].reset_index(drop=True)

    # Convert to timestamp to datetime variable
    df_wikiviews['timestamp'] = pd.to_datetime(df_wikiviews['timestamp'], format='%Y%m%d%H')

    # Rename the column from 'en.wikipedia' to 'en' and same for other languages
    df_wikiviews['project'] = df_wikiviews['project'].str.replace(r'\..*', '', regex=True)

    # Pivot the table to simplify further uses
    df_wikiviews = df_wikiviews.pivot_table(index = 'timestamp', columns = ['project'], values = 'views')

    main_url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/'

    # Name of the games you want to get the data of:
    games = ['Among Us', 'Fall Guys']

    df_gameviews = pd.DataFrame()

    for language in country_code:
        for game in games:
            try:
                url = main_url + language + '.wikipedia/all-access/user/' + game + '/daily/' + start_dt + '/' + end_dt
                r = requests.get(url,headers=headers)

                df_onequery = pd.DataFrame(r.json()['items'])
                df_gameviews = pd.concat([df_gameviews, df_onequery])

                time.sleep(0.5) # In case the IP address is blocked
            except:
                print('The {} page of {} is not found during these time'.format(language,game))
                
    # Keep only relevant columns and reset index:
    df_gameviews = df_gameviews[['project', 'article', 'timestamp','views']].reset_index(drop=True)

    # Convert timestamp to datetime format:
    df_gameviews['timestamp'] = pd.to_datetime(df_gameviews['timestamp'], format='%Y%m%d%H')

    # Rename the column from 'en.wikipedia' to 'en' and same for other languages
    df_gameviews['project'] = df_gameviews['project'].str.replace(r'\..*', '', regex=True)

    # Pivot table to have main column being the game, and subcolumns being the language`
    df_gameviews = df_gameviews.pivot_table(index = 'timestamp', columns = ['article', 'project'], values = 'views')

    # Rename columns
    df_gameviews.columns.set_names(['Game Name', 'Language'], level=[0, 1], inplace=True)

    # Filter the DataFrame to contain the data on a certain interval
    start_date = '2020-01-01'
    end_date = '2020-12-12'
    df_plot = df_gameviews.loc[start_date:end_date]

    fig, axes = plt.subplots(len(games), 1, figsize=(32, 20))

    # Loop through the subplots and create and plot a figure for each game
    for ax, game in zip(axes, games):

        fig_to_plot = return_game_figure(game, df_plot, interventions, 'views')

        sub_ax = fig_to_plot.get_axes()[0]
        sub_ax.get_figure().canvas.draw()

        buf = sub_ax.get_figure().canvas.renderer.buffer_rgba()
        plt.close(fig_to_plot)

        ax.imshow(buf)
        ax.axis('off')

    plt.xticks(rotation=45)  # Rotates the x-axis labels for better readability
    fig.tight_layout() # Adjusts the plot to ensure everything fits without overlapping
    plt.show()
    return

def save_topics_in_chunk(df_topiclinked):
    video_games_data = df_topiclinked[df_topiclinked['Culture.Media.Video games'] == True]
    game_topics = video_games_data['index'].str.replace('_',' ').values
    # Divide the whole topic datasets to implement multi-thread web-parsing in the website to increase efficiency.
    div = np.arange(0,game_topics.shape[0],4000)
    for i in range(len(div)):
        sub = game_topics[div[i]:div[i+1]-1] if i < len(div)-1 else game_topics[div[i]:]
        file_path = './game_topics/game_topic_'+str(i)+'.npy'
        if not os.path.exists(file_path):
            np.save(file_path,sub)  # Save the game topic datasets into seperated files
    return div

# save the crawled dats into the seperate files
def save_to_csv(data, num):
    file_name = f'./game_topics/game_topic_{str(num)}.csv'
    file_exists = os.path.isfile(file_name)
    with open(f'./game_topics/game_topic_{str(num)}.csv','a',newline='',encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        if not file_exists:
            csv_writer.writerow(['lang','topic','topic_in_English'])

        csv_writer.writerows(data)

# crawl the translated game topics in different languages
def crawl_title_lang(num):
    game_topics = np.load('./game_topics/game_topic_'+str(num)+'.npy',allow_pickle=True) #import the English-version game topics
    file_path = f'./game_topics/game_topic_{str(num)}.csv'
    if not os.path.exists(file_path):
        gamebar = tqdm(game_topics)

        # define the header setting for the parser
        count_dtpoint = 0
        chrome_options = Options()
        chrome_options.add_argument("User-Agent=ADABot/0.0 (floydchow7@gmail.com)")
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-extensions')
        
        # Parse the designated website
        for  game_topic in gamebar:
            title_languauge = []
            gamebar.set_description(f'Processing: {game_topic} with current title_languauge in length {str(count_dtpoint)}')
            url = 'https://pageviews.wmcloud.org/langviews/?project=en.wikipedia.org&platform=all-access&agent=user&range=latest-20&sort=views&direction=1&view=list&page='+game_topic
            English_name = game_topic
            try:
                driver = webdriver.Chrome(options=chrome_options)
                driver.get(url)
                wait = WebDriverWait(driver, 20)
                wait.until(
                    EC.visibility_of_element_located((By.ID,'output_list'))
                )
                html = driver.page_source
                driver.refresh()
                driver.delete_all_cookies()
                driver.close()
                driver.quit()

                # extract the needed elements from the website
                root = etree.HTML(html)
                names = root.xpath('//*[@id="output_list"]//tr//td//a[@dir="ltr"]//text()')
                langs = root.xpath('//*[@id="output_list"]//tr//td//a[@dir="ltr"]//@lang')

                # save the data into the files
                for lang, name in zip(langs, names):
                    title_languauge.append([lang, name, English_name])

                count_dtpoint = count_dtpoint + len(title_languauge)
                save_to_csv(title_languauge, num)
                del title_languauge
                del html
                gc.collect()


            except Exception as e:
                print(f"An error occured on {game_topic} : {(str(e)).split('Stacktrace')[0]}")
        try:
            driver.quit()
        except:
            pass
    #save_to_csv(title_languauge, num)
    #output = pd.DataFrame(title_languauge,columns=['lang','topic','topic_in_English'])
    #output.to_csv(f'./game_topics/topic_in_different_lang_{str(num)}_{str(index)}.csv')

# define the thread class for the data parsing
class titlecrawlThread(threading.Thread):
    def __init__(self, num):
        threading.Thread.__init__(self)
        self.num = num
    def run(self):
        crawl_title_lang(self.num)

def start_title_crawler_thread(div):
    thread_list = []
    for i in range(len(div)):
        thread = titlecrawlThread(i)
        thread.start()
        thread_list.append(thread)

    for thread in thread_list:
        thread.join()

# Crawl the pageviews of different datasets
def crawl_pageviews(thread_num,start_dt, end_dt):
    df_topics = pd.read_csv(f'./game_topics/game_topic_{str(thread_num)}.csv')
    eng_topics = list(set(df_topics['topic_in_English'].values))
    file_path = f'./pageviews/game_topic_{str(thread_num)}.csv'
    if not os.path.exists(file_path):
        loopbar = tqdm(eng_topics)
        headers = {'User-Agent':'ADABot/0.0 (floydchow7@gmail.com)'}
        df_wikiviews = pd.DataFrame()
        for eng_topic in loopbar:
            df_topic = df_topics[df_topics['topic_in_English']==eng_topic]
            loopbar.set_description(f"Processing {eng_topic} pageviews in {str(df_topic.shape[0])} language(s)")
            for index, row in df_topic.iterrows():
                lang = row['lang']
                topic = row['topic']
            # Declare f-string for all the different requests:
                url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/{topic}/daily/{start_dt}/{end_dt}"
            try:
                r = requests.get(url, headers=headers)
                df_onequery = pd.DataFrame(r.json()['items'])
                df_onequery['topic'] = eng_topic
                df_wikiviews = pd.concat([df_wikiviews,df_onequery])
                time.sleep(0.5) # In case the IP address is blocked
                print(f'\r{" "*100}\rThe {eng_topic} pageviews in {lang} version found', end='', flush=True)
            except:
                print(f'\r{" "*100}\rThe {eng_topic} pageviews in {lang} version NOT found', end='', flush=True)
        

    # Convert to timestamp to datetime variable
        df_wikiviews['timestamp'] = pd.to_datetime(df_wikiviews['timestamp'], format='%Y%m%d%H')

    # Rename the column from 'en.wikipedia' to 'en' and same for other languages
        df_wikiviews['lang'] = df_wikiviews['project'].str.replace(r'\..*', '', regex=True)

        df_wikiviews = df_wikiviews[['topic','lang', 'timestamp', 'views',]].reset_index(drop=True)

        df_wikiviews.to_csv(f'./pageviews/game_topic_{str(thread_num)}.csv')
    
    return 0

class pageviewcrawlThread(threading.Thread):
    def __init__(self, thread_num, start_dt, end_dt):
        threading.Thread.__init__(self)
        self.thread_num = thread_num
        self.start_dt = start_dt
        self.end_dt = end_dt
    def run(self):
        crawl_pageviews(self.thread_num,self.start_dt, self.end_dt)

def start_pageview_crawler_thread(start_dt, end_dt):
    thread_list = []
    for i in range(9):
        thread = pageviewcrawlThread(i, start_dt, end_dt)
        thread.start()
        thread_list.append(thread)

    for thread in thread_list:
        thread.join()

def crawl_uncrawled_pageviews(df_topiclinked,thread_num,langs, start_dt, end_dt):
    file_path = f'./pageviews/game_topic_{str(thread_num+1)}.csv'
    if not os.path.exists(file_path):
    # Then we try to extract the untranslated game topics.
        video_games_data = df_topiclinked[df_topiclinked['Culture.Media.Video games'] == True]
        game_topics = video_games_data['index'].str.replace('_',' ').values
        uncrawled_topics = set(game_topics)
        for i in range(thread_num + 1):
            df_topics = pd.read_csv(f'./game_topics/game_topic_{str(i)}.csv')
            crawled_topics = set(df_topics['topic_in_English'].values)
            uncrawled_topics = uncrawled_topics - crawled_topics
        # Then we try to extract uncrawled_topic and form posudo-api links since we didn't know the actual translation in different languages
        print(f"There is {str(len(uncrawled_topics))} topics need to be crawled")
        headers = {'User-Agent':'ADABot/0.0 (floydchow7@gmail.com)'}
        df_wikiviews = pd.DataFrame()
        
        loopbar = tqdm(list(uncrawled_topics))
        for uncrawled_topic in loopbar:
            loopbar.set_description(f"Processing {uncrawled_topic} pageviews")
            for lang in langs:        
                url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/{uncrawled_topic}/daily/{start_dt}/{end_dt}"
                try:
                    r = requests.get(url, headers=headers)
                    df_onequery = pd.DataFrame(r.json()['items'])
                    df_onequery['topic'] = uncrawled_topic
                    df_wikiviews = pd.concat([df_wikiviews,df_onequery])
                    time.sleep(0.5) # In case the IP address is blocked
                    print(f'\r{" "*100}\rThe {uncrawled_topic} pageviews in {lang} version found', end='', flush=True)
                except:
                    pass

            # Convert to timestamp to datetime variable
        df_wikiviews['timestamp'] = pd.to_datetime(df_wikiviews['timestamp'], format='%Y%m%d%H')

        # Rename the column from 'en.wikipedia' to 'en' and same for other languages
        df_wikiviews['lang'] = df_wikiviews['project'].str.replace(r'\..*', '', regex=True)

        df_wikiviews = df_wikiviews[['topic','lang', 'timestamp', 'views',]].reset_index(drop=True)

        df_wikiviews.to_csv(f'./pageviews/game_topic_{str(thread_num+1)}.csv')
    
# Now we try to extract the categories for each wikidata
def extract_game_genre(thread_num):
    file_path = f'./game_genres/game_genres_{str(thread_num)}.csv'
    if not os.path.exists(file_path):
        raw_gametopic_df = pd.read_csv('./game_topics/raw_gametopic_data.csv')
        game_topic_df = pd.read_csv(f'./pageviews/game_topic_{str(thread_num)}.csv')
        game_topic_df = set(game_topic_df['topic'])
        selected_gametopic_df = raw_gametopic_df[raw_gametopic_df['index'].isin(game_topic_df)].copy()
        selected_gametopic_df['genres'] = pd.NA
        # Define url for query
        endpoint_url = "https://query.wikidata.org/sparql"
        headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json'
            }
        for index, row in tqdm(selected_gametopic_df.iterrows(), total=len(selected_gametopic_df), desc="Processing rows"):
            qid = row['qid']
            query = """
            SELECT ?genreLabel
            WHERE {
                wd:""" + qid + """ wdt:P136 ?genre.
                SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
            }
            """
            response = requests.get(endpoint_url, params={'query': query, 'format': 'json'}, headers=headers)
            data = response.json()
            # extract the genres in the wikipidea pages for every game topic
            genres = [item['genreLabel']['value'] for item in data['results']['bindings']] if 'results' in data else []
            selected_gametopic_df.at[index, 'genres'] = genres
            time.sleep(0.5)
        selected_gametopic_df.to_csv(f'./game_genres/game_genres_{str(thread_num)}.csv',index=False,encoding='utf-8')
    return None

class genrecrawlThread(threading.Thread):
    def __init__(self, thread_num):
        threading.Thread.__init__(self)
        self.thread_num = thread_num
    def run(self):
        extract_game_genre(self.thread_num)

def start_genre_crawler_thread():
    thread_list = []
    for i in range(10):
        thread = genrecrawlThread(i)
        thread.start()
        thread_list.append(thread)

    for thread in thread_list:
        thread.join()

def filter_game_genres(raw_game_filepath):
    game_df = pd.read_csv(raw_game_filepath)
    # We only extract the topics that are actually games, which will have the unempty genres columns
    df = game_df.copy(deep=True)
    df['genres'].apply(lambda x: len(x)>2)
    df = df.loc[df['genres'].apply(lambda x: len(x)>2),['index','genres']]
    df['genres'] = df['genres'].apply(lambda x: x.replace("'","").replace("\"","").replace("[","").replace("]","").split(","))
    new_df = pd.DataFrame(columns=['index','genres'])

    # Split the multiple genres into different rows
    for index, row in df.iterrows():
        game = row['index']
        genre_list = row['genres']
        for genre in genre_list:
            new_df.loc[len(new_df.index)] = [game, genre]

    # We check whether there is a tab in the genres columns and convert it to normal one
    new_df['genres'] = new_df['genres'].apply(lambda x: x[1:] if x[0]==' 'else x)

    # count the games in different genres
    count_genres_df = new_df.groupby(['genres'],as_index=False).agg({'index':'count'}).sort_values('index',ascending=False).reset_index(drop=True)
    count_genres_df.columns = ['genres','count']

    # Then we aggergate the count_df with the genre_df to obtain the main genres(which means the highest genres) in the datasets
    new_df2 = pd.merge(new_df,count_genres_df,on='genres',how='left')

    # rank the raw game genres
    tmp_df = new_df2.groupby('index',as_index=False).apply(lambda x: x.sort_values(by='count',ascending=False))[['index','genres','count']].reset_index(drop=True)
    tmp_df['rank'] = tmp_df.groupby('index').cumcount() + 1
    tmp_df['rank'] = tmp_df['rank'].apply(lambda x: 'genre '+ str(x))
    
    # alternate the dataset into the pivot table
    tmp_pivot_df = tmp_df.pivot(index='index',columns='rank',values='genres').reset_index()
    reorder_columns = ['genre '+str(i) for i in np.arange(1,14,1)]
    reorder_columns.insert(0,'index')
    tmp_pivot_df = tmp_pivot_df[reorder_columns]

    # We only obtain the genres with the highest count as the main genres
    result_df = new_df2.copy(deep=True)
    return result_df

def main_genre_classification(raw_classification_filepath, result_df):
    gpt_classification_df = pd.read_csv(raw_classification_filepath)
    gpt_classification_df = gpt_classification_df[['Small Game Genres','Larger Game Genres']]
    gpt_classification_df['Small Game Genres'] = gpt_classification_df['Small Game Genres'].apply(lambda x: x.lower().replace("'",""))
    gpt_classification_df['Larger Game Genres'] = gpt_classification_df['Larger Game Genres'].apply(lambda x: x.split(",")[0])

    #Explore the chatGPT-classification in the main Genres
    gpt_classification_df.columns = ['genres','Main Genre']
    main_genre_df = pd.merge(result_df, gpt_classification_df, on='genres', how='left')

    main_genre_df = main_genre_df[['index','Main Genre','genres']]
    main_genre_df.columns = ['Game','Main Genre','Secondary Genre']

    main_genre_df.to_csv('./Milestone3/gpt-classification.csv',encoding='utf-8',index=False)
    return main_genre_df

def display_main_genre_stats(main_genre_df):
    stats_df = main_genre_df.drop_duplicates(subset=['Game','Main Genre']).groupby("Main Genre",as_index=False).agg({"Game":"count"}).sort_values("Game",ascending=False).reset_index(drop=True)
    return stats_df

def visualize_genres_distribution(stats_df, others_threadshold):
    # Create a new DataFrame for 'Others' category
    df_others = stats_df[stats_df['Game'] <= others_threadshold]
    others_row = pd.DataFrame({'Main Genre': ['Others'], 'Game': [df_others['Game'].sum()]})
    stats_df = pd.concat([stats_df[stats_df['Game'] > others_threadshold], others_row], ignore_index=True)

    # Sort the DataFrame by 'Game' column in descending order
    stats_df = stats_df.sort_values(by='Game', ascending=True)

    # Plot using Plotly Express
    fig = px.bar(stats_df, x='Game', y='Main Genre', orientation='h', color='Game',
                 labels={'Game': 'Number of Games', 'Main Genre': 'Main Genre'},
                 title='Game Genres Distribution',
                 text='Game', height=500)

    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False)

    html_file_path = "game genre count.html"
    pio.write_html(fig, file=html_file_path)

    # Show the plot
    fig.show()

def visualize_pageviews_in_genre(pageviews_filepath, genres_filepath):
    pageviews = pd.read_csv(pageviews_filepath)
    game_genres = pd.read_csv(genres_filepath)
    pageviews.columns = ['Game', 'lang', 'timestamp', 'views']
    merged_df = pd.merge(pageviews, game_genres, on='Game', how='left')
    merged_df.dropna(inplace=True)
    grouped_df = merged_df.groupby(by=['Main Genre', 'timestamp', 'lang'], as_index=False).agg(pageviews=pd.NamedAgg(column='views', aggfunc='sum'))

    # We visualize the total pageviews according to the game genres on some main languages except English
    main_genres = list(set(grouped_df['Main Genre']))
    main_genres.remove('Comics')

    # Create a single subplot
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Game Genres Pageviews'], shared_xaxes=True, shared_yaxes=True)

    # Create traces for each genre
    traces = []

    for genre in main_genres:
        sub_grouped_df = grouped_df[(grouped_df['Main Genre'] == genre) & (grouped_df['lang'].isin(['de', 'fr', 'it', 'pt', 'es', 'ja']))]
        sub_grouped_df = sub_grouped_df.copy()
        sub_grouped_df['timestamp'] = pd.to_datetime(sub_grouped_df['timestamp'])

        for lang in sub_grouped_df['lang'].unique():
            lang_data = sub_grouped_df[sub_grouped_df['lang'] == lang]
            trace = go.Scatter(x=lang_data['timestamp'], y=lang_data['pageviews'], mode='lines', name=f'{genre} | {lang}', showlegend=True)
            traces.append(trace)
            fig.add_trace(trace)

        # Add a hidden trace for each genre
        #hidden_trace = go.Scatter(x=lang_data['timestamp'], y=np.zeros(len(lang_data['pageviews'])), name='', showlegend=True)
        #traces.append(hidden_trace)
        #fig.add_trace(hidden_trace)

    # Customize layout
    fig.update_layout(title_text='Game Genres Pageviews', height=600, legend=dict(x=-0.3, y=0.5, font=dict(size=8)))

    # Add dropdown menu to select different main genres
    dropdown_buttons = [
        {'label': genre, 'method': 'update', 'args': [{'visible': [genre == trace.name.split(' | ')[0] for trace in traces]}]}
        for genre in main_genres
    ]

    fig.update_layout(updatemenus=[{'active': 0, 'buttons': dropdown_buttons, 'showactive': True}])


    html_file_path = f"pageviews.html"
    pio.write_html(fig, file=html_file_path)
    # Show the plot
    fig.show()

def convert_to_code_dict(df_code):
    #convert it the dictionary
    code_dict = dict(zip(df_code['lang'],df_code['state']))
    return code_dict

def merge_mobility_pageview(globalmob, pageviews, game_genres, code_dict):

    #Align the pageviews and categories
    pageviews.columns = ['Game','lang','timestamp','views']
    merged_df = pd.merge(pageviews, game_genres,on='Game',how='left')
    merged_df.dropna(inplace=True)
    grouped_df = merged_df.groupby(by=['Main Genre','timestamp','lang'],as_index=False).agg(pageviews = pd.NamedAgg(column='views',aggfunc='sum'))
    grouped_df = grouped_df.replace({'lang': code_dict})
    lang_pageviews_df = grouped_df.groupby(by=['lang','timestamp'],as_index=False).agg(pageviews = pd.NamedAgg(column='pageviews',aggfunc='sum'))
    lang_pageviews_df['lang'] = lang_pageviews_df['lang'].apply(lambda x: x.upper())
    lang_pageviews_df.columns = ['country_region_code','date','pageviews']
    # We change pageviews to baseline change
    baseline = '2020-02-14' #Define it as the baseline for the pageviews
    lang_pageviews_df = lang_pageviews_df[lang_pageviews_df['date']>=baseline]

    baseline_pageviews = lang_pageviews_df[lang_pageviews_df['date']==baseline]
    baseline_pageviews.columns = ['country_region_code','date','baseline pageviews']
    baseline_pageviews.drop(['date'], axis=1,inplace=True)
    lang_pageviews_df = pd.merge(lang_pageviews_df,baseline_pageviews,on='country_region_code',how='left')
    lang_pageviews_df['change from baseline'] = 100*(lang_pageviews_df['pageviews']/lang_pageviews_df['baseline pageviews']-1) # Calculate the change compared to baseline pageview in different languages

    globalmob = pd.merge(globalmob, lang_pageviews_df,on=['country_region_code','date'],how='left') #Merged with global mobility datasets
    globalmob.drop(['pageviews','baseline pageviews'],axis=1, inplace=True)
    return grouped_df, globalmob

def visualize_mobility_pageviews(globalmob, interventions, df_code):
    # Create a subplot with Plotly
    fig = make_subplots(rows=(len(df_code['lang'])//2)+1, cols=2, shared_yaxes=False, subplot_titles=df_code.index.tolist(), vertical_spacing=0.1)

    for i, c in enumerate(df_code['lang']):
        cs = df_code.iloc[i]['state']

        if cs == 'KR':
            globalmob_ko = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['sub_region_1'].isnull()) & (globalmob['metro_area'].isnull())]
        else:
            if cs == 'RS':
                globalmob_sr = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['metro_area'].isnull())]
            else:
                if cs == 'ES':
                    globalmob_ca = globalmob[(globalmob['country_region_code'] == 'ES') & (globalmob['sub_region_1'] == 'Catalonia') & (globalmob['sub_region_2'].isnull())].copy()
                else:
                    globalmob_g = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['sub_region_1'].isnull())].copy()
                    globalmob_g.reset_index(inplace=True, drop=True)

        df = globalmob_g.copy(deep=True)

        mobility_g = interventions.loc[c]['Mobility']
        lockdown_g = interventions.loc[c]['Lockdown']
        normalcy_g = interventions.loc[c]['Normalcy']

        columns = globalmob.columns[8:]
        df = df.drop(['residential_percent_change_from_baseline', 'parks_percent_change_from_baseline'], axis=1)
        columns = columns.drop(['residential_percent_change_from_baseline', 'parks_percent_change_from_baseline'])

        mean_g = df[columns.drop(['change from baseline'])].mean(axis=1)

        row = i // 2 + 1
        col = i % 2 + 1

        # Create traces for the plot
        mean_trace = go.Scatter(x=df['date'], y=mean_g, mode='lines', name='Average change in mobility', line=dict(color='blue'))
        pageview_trace = go.Scatter(x=df['date'], y=df['change from baseline'], mode='lines', name='Change in pageviews in Games', line=dict(color='red'))

        # Add traces to the subplot
        fig.add_trace(mean_trace, row=row, col=col)
        fig.add_trace(pageview_trace, row=row, col=col)


        # Add vertical lines for events if not NA
        if not pd.isna(lockdown_g):
            fig.add_shape(type="line", x0=lockdown_g, x1=lockdown_g, y0=-100, y1=200, line=dict(color="black", width=2.2, dash="dash"), row=row, col=col)

        if not pd.isna(mobility_g):
            fig.add_shape(type="line", x0=mobility_g, x1=mobility_g, y0=-100, y1=200, line=dict(color="blue", width=1.5, dash="solid"), row=row, col=col)

        if not pd.isna(normalcy_g):
            fig.add_shape(type="line", x0=normalcy_g, x1=normalcy_g, y0=-100, y1=200, line=dict(color="black", width=1.5, dash="solid"), row=row, col=col)


        # Update x-axis ticks
        tickvals = [13, 42, 73, 103, 134, 164]
        ticktext = ['28 Feb', '28 Mar', '28 Apr', '28 May', '28 Jun', '28 Jul']

        #fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, row=row, col=col)


    # Show the plot
    fig.update_layout(title_text="Change of mobility and game pageviews in different countries in Year 2020(%)", showlegend=False)
    fig.update_layout(height=1200, width=1000)  # Adjust the figure size as needed
    html_file_path = "pageviews and countries.html"
    pio.write_html(fig, file=html_file_path)

    fig.show()

def visualize_different_language(lang, baseline, interventions, grouped_df, globalmob, code_dict, country_name, omit_genre=[]):
    # Extract data
    pageviews_sub = grouped_df[(grouped_df['lang'] == lang.upper()) & (grouped_df['timestamp'] >= baseline)]
    baseline_pageviews_sub = pageviews_sub[pageviews_sub['timestamp'] == baseline].drop(['lang', 'timestamp'], axis=1)
    pageviews_sub.columns = ['Main Genre', 'date', 'lang', 'pageviews']
    baseline_pageviews_sub.columns = ['Main Genre', 'baseline pageviews']
    pageviews_sub = pd.merge(pageviews_sub, baseline_pageviews_sub, on=['Main Genre'], how='left')
    pageviews_sub['change from basetime'] = 100 * (pageviews_sub['pageviews'] / pageviews_sub['baseline pageviews'] - 1)
    pageviews_sub['lang'] = pageviews_sub['lang'].apply(lambda x: x.upper())
    pageviews_sub_result = pd.pivot_table(pageviews_sub, values='change from basetime', index='date',
                                          columns='Main Genre').reset_index().dropna(axis=1)

    globalmob_g = globalmob[(globalmob['country_region_code'] == lang.upper()) & (globalmob['sub_region_1'].isnull())].drop(
        ['change from baseline'], axis=1).dropna(axis=1)

    df = pd.merge(globalmob_g, pageviews_sub_result, on='date', how='left')

    matching_lang = [key for key, value in code_dict.items() if value == lang.upper()]

    mobility_fr = interventions.loc[matching_lang[0]]['Mobility']
    lockdown_fr = interventions.loc[matching_lang[0]]['Lockdown']
    normalcy_fr = interventions.loc[matching_lang[0]]['Normalcy']

    columns = df.columns[4:].copy()

    mean_fr = df.loc[:, columns.drop(pageviews_sub_result.columns.drop(['date']))].mean(axis=1)

    selected_genres = pageviews_sub_result.columns.drop(['date'] + omit_genre) if len(omit_genre) > 0 else pageviews_sub_result.columns.drop(['date'])

    # Create Plotly subplot
    fig = make_subplots(rows=(len(selected_genres) + 1) // 2, cols=2,
                        subplot_titles=list(selected_genres))
    fig.update_annotations(font_size=10)
    fig.update_layout(font=dict(size=8))
    fig.update_layout(margin=dict(l=0,r=6,b=4), mapbox_style = "open-street-map")
    for i, genre in enumerate(selected_genres):
        row = i // 2 + 1
        col = i % 2 + 1
        genre_trace = go.Scatter(x=df['date'], y=df[genre], mode='lines', name=genre, line=dict(color='red',width=1))
        fig.add_trace(genre_trace, row=row, col=col)

        mean_trace = go.Scatter(x=df['date'], y=mean_fr, mode='lines', name='Average percentage change in mobility' , line=dict(color='blue',width=1))
        fig.add_trace(mean_trace, row=row, col=col)
    # Add vertical lines for events if not NA
        if not pd.isna(lockdown_fr):
            lockdown_shape = dict(type="line", x0=lockdown_fr, x1=lockdown_fr, y0=-100,y1=max(df[genre]),
                                line=dict(color="black", width=1, dash="dash"))
            fig.add_shape(lockdown_shape, row=row, col=col)

        if not pd.isna(mobility_fr):
            mobility_shape = dict(type="line", x0=mobility_fr, x1=mobility_fr, y0=-100,y1=max(df[genre]),
                                line=dict(color="blue", width=1, dash="solid"))
            fig.add_shape(mobility_shape, row=row, col=col)

        if not pd.isna(normalcy_fr):
            normalcy_shape = dict(type="line", x0=normalcy_fr, x1=normalcy_fr, y0=-100,y1=max(df[genre]),
                                line=dict(color="black", width=1, dash="solid"))
            fig.add_shape(normalcy_shape, row=row, col=col)


    # Update layout
    fig.update_layout(
        title_text=f'Mobility and attention shift in different game genres in {country_name}(%)',
        showlegend=False,
        height=800, 
        width=500, 
    )

    # Save the figure as an HTML file
    html_file_path = f"{country_name.lower()}.html"
    pio.write_html(fig, file=html_file_path)

    # Display the HTML link
    fig.show()
    # Define the function to visualize the pageviews and mobilities change in different game genres
    """_summary_

    Args:
        lang : the language we analyze
        grouped_df : the pageviews datasets grouped from game genres
        globalmob : the global mobilites function
        code_dict (_type_): country code dictionary
        omit_genre (optional): The game genre to omit during analysis
    """
    pageviews_sub = grouped_df[(grouped_df['lang']==lang.upper())&(grouped_df['timestamp']>=baseline)]
    baseline_pageviews_sub = pageviews_sub[pageviews_sub['timestamp']==baseline].drop(['lang','timestamp'],axis=1)
    pageviews_sub.columns=['Main Genre','date','lang','pageviews']
    baseline_pageviews_sub.columns = ['Main Genre','baseline pageviews']
    pageviews_sub = pd.merge(pageviews_sub, baseline_pageviews_sub,on=['Main Genre'],how='left')
    pageviews_sub['change from basetime'] = 100*(pageviews_sub['pageviews']/pageviews_sub['baseline pageviews']-1)
    pageviews_sub['lang'] = pageviews_sub['lang'].apply(lambda x: x.upper())
    pageviews_sub_result = pd.pivot_table(pageviews_sub,values='change from basetime', index='date',columns='Main Genre').reset_index().dropna(axis=1)

    globalmob_g = globalmob[(globalmob['country_region_code'] == lang.upper()) & (globalmob['sub_region_1'].isnull())].drop(['change from baseline'],axis=1).dropna(axis=1)

    df = pd.merge(globalmob_g, pageviews_sub_result, on='date',how='left')

    #selected_genres = ['Action', 'Adult',
    #    'Adventure', 'Anime/Manga', 'Fantasy', 'Horror',
    #    'Multiplayer/Online', 'Puzzle', 'Racing',
    #        'Sports', 'Strategy']
    matching_lang = [key for key, value in code_dict.items() if value == lang.upper()]

    mobility_fr = interventions.loc[matching_lang[0]]['Mobility']
    lockdown_fr = interventions.loc[matching_lang[0]]['Lockdown']
    normalcy_fr = interventions.loc[matching_lang[0]]['Normalcy']

    columns = df.columns[4:].copy()

    mean_fr = df.loc[:, columns.drop(pageviews_sub_result.columns.drop(['date']))].mean(axis=1)

    selected_genres = pageviews_sub_result.columns.drop(['date']+ omit_genre) if len(omit_genre)>0 else pageviews_sub_result.columns.drop(['date'])

    #fig, axs = plt.subplots(len(pageviews_sub_result.columns.drop(['date']))//2, 2, sharey=True, figsize=(20, 20))
    fig, axs = plt.subplots((len(selected_genres)+1)//2, 2, sharey=True, figsize=(20, 20))

    for i, genre in enumerate(selected_genres):

        row = i // 2
        col = i % 2
        ax = axs[row, col]
        mean_line, = ax.plot(df['date'], mean_fr, label='Average percentage change in mobility')
        for column in columns:
            if column in pageviews_sub_result.columns:
                if column in [genre]:
                    genre_line, = ax.plot(df['date'], df[column], label=column,color='red',linestyle='--')
                elif column not in omit_genre:
                    ax.plot(df['date'], df[column], label=column, color='red',linestyle='--', alpha=0.1)
                else:
                    pass
            else:
                ax.plot(df['date'], df[column], label=column, color='black', alpha=0.1)

        ax.axvline(lockdown_fr, color='black', lw=2.2, linestyle="--")
        ax.axvline(mobility_fr, color='blue', lw=1.5, linestyle="-", alpha=0.7)
        ax.axvline(normalcy_fr, color='black', lw=1.5, linestyle="-", alpha=0.5)
        ax.set_xticks([13, 42, 73, 103, 134, 164], ['28 Feb', '28 Mar', '28 Apr', '28 May', '28 Jun', '28 Jul'])
        ax.set_xlim(min(df['date']), max(df['date']))
        ax.grid(True)
        ax.legend(handles=[mean_line,genre_line],loc='upper right')
    plt.suptitle(f'Mobility and attention shift in different game genres in {country_name}',fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plt.show()

def defglobalmob(cs, globalmob):
    match cs:

        case 'KR':
          globalmob_ko = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['sub_region_1'].isnull()) & (globalmob['metro_area'].isnull())].copy()
          df = globalmob_ko

        case 'RS':
          globalmob_sr = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['metro_area'].isnull())].copy()
          globalmob_sr.reset_index(inplace=True)
          globalmob_sr.drop('index', axis=1, inplace=True)

          date_range = pd.date_range('2020-05-19', '2020-07-02', normalize=True)
          date_range = date_range.date
          date_range_str = [d.strftime("%Y-%m-%d") for d in date_range]

          # Rows to be inserted
          num_rows_to_insert = 45
          new_rows_data = {'country_region_code': 'RS',
                          'country_region': 'Serbia',
                          'sub_region_1': [np.nan] * num_rows_to_insert,
                          'sub_region_2': [np.nan] * num_rows_to_insert,
                          'metro_area': [np.nan] * num_rows_to_insert,
                          'iso_3166_2_code': [np.nan] * num_rows_to_insert,
                          'census_fips_code': [np.nan] * num_rows_to_insert,
                          'date': date_range_str,
                          'retail_and_recreation_percent_change_from_baseline': -15.25,
                          'grocery_and_pharmacy_percent_change_from_baseline': -15.25,
                          'transit_stations_percent_change_from_baseline': -15.25,
                          'workplaces_percent_change_from_baseline': -15.25
          }
          new_rows_df = pd.DataFrame(new_rows_data)

          # Index where you want to insert the new rows (in this case, after the second row)
          index_to_insert = 93

          # Insert the new rows
          merged = pd.concat([globalmob_sr.loc[:index_to_insert], new_rows_df, globalmob_sr.loc[index_to_insert+1:]]).reset_index(drop=True)
          globalmob_sr = merged
          df = merged

        case 'ES':
          globalmob_ca = globalmob[(globalmob['country_region_code'] == 'ES') & (globalmob['sub_region_1'] == 'Catalonia') & (globalmob['sub_region_2'].isnull())].copy()
          df = globalmob_ca

        # If an exact match is not confirmed, this last case will be used if provided
        case _:
          globalmob_g = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['sub_region_1'].isnull())].copy()
          globalmob_g.reset_index(inplace=True, drop=True)
          df = globalmob_g.copy(deep=True)

    df.drop(['residential_percent_change_from_baseline', 'parks_percent_change_from_baseline'], axis=1, inplace=True)
    df = df.reset_index(drop=True)
    columns = df.columns[8:]
    mean_g = df[columns].mean(axis=1)
    mean_g = mean_g.reset_index(drop=True)
    return df, mean_g

def plotmob_inter(globalmob, df_code, interventions):
  fig = make_subplots(rows=6, cols=2, shared_xaxes=True, shared_yaxes=True,
                          horizontal_spacing = 0.025, vertical_spacing = 0.025,
                          subplot_titles=("Mobility in " + df_code.index))

  for i, c in enumerate(df_code['lang']):
      cs = df_code.iloc[i]['state']
      df, mean_g = defglobalmob(cs, globalmob)

      mobility_g = interventions.loc[c]['Mobility']
      lockdown_g = interventions.loc[c]['Lockdown']
      normalcy_g = interventions.loc[c]['Normalcy']

      x = df['date'].index
      y = mean_g
      poly = np.polyfit(x, y, 10)
      poly_y = np.poly1d(poly)(x)


      row = i // 2 +1
      col = i % 2 +1

      n = "Approximation for " + df_code.index[i]
      s = False
      if cs == 'JP':
        s = True

        # Add vertical lines at x = 2 and x = 4
      fig.add_trace(go.Scatter(x=df['date'], y=mean_g, mode='lines', name='Line 1', line=dict(color='rgba(0,0,205,1)', width=1), showlegend = False), row=row, col=col)
      fig.add_trace(go.Scatter(x=df['date'], y=poly_y, mode='lines', line=dict(color='rgba(255,140,0,1)', width=1),
                               hoverinfo='skip', legendgroup="group", name= 'Approximation curves',
                               showlegend = s, visible='legendonly'), row=row, col=col)
      for column in df.columns[8:]:
          fig.add_trace(go.Scatter(x=df['date'], y=df[column], mode='lines', line=dict(color='rgba(0,0,0,0.1)', width=1),
                                   hoverinfo='skip', showlegend = False), row=row, col=col)

      fig.update_annotations(font_size=12)
      fig.update_layout(font=dict(size=8))
      fig.update_yaxes(showgrid=True, range=[-100, 50], dtick=20, title_text='change (%)', row=row, col=col)
      fig.update_xaxes(title_text='Date', row=row, col=col)
      fig.add_shape(type='line', x0=mobility_g, x1=mobility_g, y0=-100, y1=50, line=dict(color='red', width=1.5), row=row, col=col)
      fig.add_shape(type='line', x0=normalcy_g, x1=normalcy_g, y0=-100, y1=50 , line=dict(color='red', width=1.5, dash='dot'), row=row, col=col)

  fig.update_layout(
    margin=dict(l=10, r=10, t=50, b=0),
    legend=dict(x=0, y=-0.1, traceorder='normal', orientation='h'),
    )

  fig.show()

def plotmob(globalmob, df_code, interventions):
  fig, axs = plt.subplots((len(interventions)-1)//2, 2, sharey=True, figsize=(20, 20))

  for i, c in enumerate(df_code['lang']):
      cs = df_code.iloc[i]['state']
      df, mean_g = defglobalmob(cs, globalmob)

      mobility_g = interventions.loc[c]['Mobility']
      lockdown_g = interventions.loc[c]['Lockdown']
      normalcy_g = interventions.loc[c]['Normalcy']

      x = df['date'].index
      y = mean_g
      poly = np.polyfit(x, y, 10)
      poly_y = np.poly1d(poly)(x)


      row = i // 2
      col = i % 2

      axs[row, col].plot(df['date'], mean_g)
      axs[row, col].plot(x, poly_y)
      for column in df.columns[8:]:
          axs[row, col].plot(df['date'], df[column], label=column, color='black', alpha=0.1)

      axs[row, col].axvline(lockdown_g, color='black', lw=2.2, linestyle="--")
      axs[row, col].axvline(mobility_g, color='blue', lw=1.5, linestyle="-", alpha=0.7)
      axs[row, col].axvline(normalcy_g, color='black', lw=1.5, linestyle="-", alpha=0.5)

      axs[row, col].set_xticks([13, 42, 73, 103, 134, 164], ['28 Feb', '28 Mar', '28 Apr', '28 May', '28 Jun', '28 Jul'])
      axs[row, col].set_xlim(min(df['date']), max(df['date']))
      axs[row, col].grid(True)
      axs[row, col].set_title('Mobility in ' + df_code.index[i])
      axs[row, col].set_xlabel('date')
      axs[row, col].set_ylabel('percentage of mobility compared to day 0')

      lines = [
        Line2D([0], [0], color='gray', alpha=0.5, linestyle='-'),
        Line2D([0], [0], color='tab:blue', linewidth=2, linestyle='-'),
        Line2D([0], [0], color='blue', linewidth=1.5, linestyle='-'),
        Line2D([0], [0], color='black', lw=2.2, linestyle="--"),
        Line2D([0], [0], color='gray', linewidth=1.5, linestyle='-')
        ]
      lines_labels = ['Mobility Signals',
                'Mean for each country',
                'Mobility change point',
                'Start of the lockdown',
                'Normalcy date'
                ]
      fig.legend(lines, lines_labels, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.025), frameon=False, fontsize=12)


  plt.tight_layout()
  plt.show()

def meanmob(df_code, globalmob):
  df_mean = pd.DataFrame()

  for i, c in enumerate(df_code['lang']):
    cs = df_code.iloc[i]['state']
    _,mean_g = defglobalmob(cs, globalmob)
    df_mean[df_code.index[i]] = mean_g

  return df_mean

def meanmobplot(df_code, interventions, globalmob):
  df_mean = meanmob(df_code, globalmob)
  df_code['mean_mob'] = None

  for i, c in enumerate(df_code['lang']):

    cs = df_code.iloc[i]['state']
    country = df_code.index[i]
    m = interventions.loc[c]['Mobility']
    n = interventions.loc[c]['Normalcy']
    mob, _ = defglobalmob(cs, globalmob)
    index_m = (mob['date'] == m).index[mob['date'] == m].tolist()[0]
    index_n = (mob['date'] == n).index[mob['date'] == n].tolist()[0]

    mean_value = df_mean[country].iloc[index_m:index_n+1].mean()
    df_code.loc[country]['mean_mob'] = mean_value

  ax = df_code['mean_mob'].plot.bar(figsize=(7, 6))
  ax.grid()  # grid lines
  ax.set_axisbelow(True)

  # Add overall title
  plt.title('Average decrease in mobility depending on the country')

  # Add axis titles
  plt.xlabel('country/region')
  plt.xticks(rotation=45);
  plt.ylabel('Average decrease in mobility [%]')
  plt.grid(True)
  plt.gca().invert_yaxis()

  # Show the plot
  plt.show()

def smoothedmobility(df_code, globalmob):
  fig, axs = plt.subplots(len(df_code)//2, 2, sharex = True, sharey = True, figsize=(20, 20))

  for i, c in enumerate(df_code['lang']):

    country_code = df_code.iloc[i]['state']
    country,av = defglobalmob(country_code, globalmob)

    x = country['date'].index
    y = av
    poly = np.polyfit(x, y, 10)
    poly_y = np.poly1d(poly)(x)

    row = i // 2
    col = i % 2
    axs[row, col].plot(x, y)
    axs[row, col].plot(x, poly_y)

    axs[row, col].set_xticks([13, 42, 73, 103, 134, 164])
    axs[row, col].set_xticklabels(['28 Feb', '28 Mar', '28 Apr', '28 May', '28 Jun', '28 Jul'])

    axs[row, col].set_xlim(min(x), max(x))

    axs[row, col].grid(True)
    axs[row, col].set_title('Mobility in ' + df_code.index[i])
    axs[row, col].set_xlabel('date')
    axs[row, col].set_ylabel('percentage of mobility compared to day 0')

  plt.tight_layout()
  plt.show()

def aggregatedmobilitysmoothedplot(df_code, globalmob):
  for i, c in enumerate(df_code['lang']):

    country_code = df_code.iloc[i]['state']
    _,av = defglobalmob(country_code, globalmob)
    country,_ = defglobalmob(country_code, globalmob)

    x = country['date'].index
    y = av
    poly = np.polyfit(x, y, 10)
    poly_y = np.poly1d(poly)(x)
    plt.plot(x, poly_y, label=c)

  plt.xticks([13, 42, 73, 103, 134, 164], ['28 Feb', '28 Mar', '28 Apr', '28 May', '28 Jun', '28 Jul'])
  plt.xlim(min(x), max(x))

  plt.grid(True)
  plt.title('Mobility depending on the country')
  plt.xlabel('date')
  plt.ylabel('percentage of mobility compared to day 0')

  plt.legend()
  plt.show()

def desaggregatedmobilitysmoothed(df_code, globalmob):
  fig, axs = plt.subplots(1, 3, sharex = True, sharey = True, figsize=(12,5))

  for i, c in enumerate(df_code['lang']):

    country_code = df_code.iloc[i]['state']
    _,av = defglobalmob(country_code, globalmob)
    country,_ = defglobalmob(country_code, globalmob)

    x = country['date'].index
    y = av
    poly = np.polyfit(x, y, 10)
    poly_y = np.poly1d(poly)(x)

    row=0
    if c in ['fr', 'ca', 'it', 'sr']:
      col = 0
    else:
      if c in ['ko', 'ja', 'sv']:
        col = 2
      else:
        col = 1

    axs[col].plot(x, poly_y, label=c)

    axs[col].set_xticks([13, 42, 73, 103, 134, 164])
    axs[col].set_xticklabels(['28 Feb', '28 Mar', '28 Apr', '28 May', '28 Jun', '28 Jul'])

    axs[col].set_xlim(min(x), max(x))

    axs[col].grid(True)
    axs[col].set_xlabel('date')
    axs[col].set_ylabel('percentage of mobility compared to day 0')

    axs[col].legend()

  axs[0].set_title('Mobility for a very restrictive lockdown')
  axs[1].set_title('Mobility for a restrictive lockdown')
  axs[2].set_title('Mobility for an unrestrictive lockdown')
  plt.legend()
  plt.tight_layout()
  plt.show()

def applemean(mobcountry_walking, mobcountry_transit, mobcountry_driving, country, interventions):

  if country == 'Korea':
    walking = mobcountry_walking[mobcountry_walking['region'].eq('Republic of Korea')]
    driving = mobcountry_driving[mobcountry_driving['region'].eq('Republic of Korea')]
    transiting = mobcountry_transit[mobcountry_transit['region'].eq('Republic of Korea')]
  else:
    if country == 'Catalonia':
      walking = mobcountry_walking[mobcountry_walking['region'].eq('Barcelona')]
      driving = mobcountry_driving[mobcountry_driving['region'].eq('Barcelona')]
      transiting = mobcountry_transit[mobcountry_transit['region'].eq('Barcelona')]
    else:
      walking = mobcountry_walking[mobcountry_walking['region'].eq(country)]
      driving = mobcountry_driving[mobcountry_driving['region'].eq(country)]
      transiting = mobcountry_transit[mobcountry_transit['region'].eq(country)]

  df = pd.concat([walking, driving, transiting])
  df_mean = df.drop(columns=['geo_type', 'region', 'transportation_type']).mean(axis=0)

  return df, df_mean

def plotmobapple(applemob, df_code, interventions):

  mobcountry_walking = applemob[applemob['transportation_type'] == 'walking']
  mobcountry_transit = applemob[applemob['transportation_type'] == 'transit']
  mobcountry_driving = applemob[applemob['transportation_type'] == 'driving']

  fig, axs = plt.subplots(len(df_code)//2, 2, sharex = True, sharey = True, figsize=(20, 20))

  for i, country in enumerate(df_code.index):

    df, df_mean = applemean(mobcountry_walking, mobcountry_transit, mobcountry_driving, country, interventions)
    c = df_code.iloc[i]['lang']

    mobility_g = interventions.loc[c]['Mobility']
    lockdown_g = interventions.loc[c]['Lockdown']
    normalcy_g = interventions.loc[c]['Normalcy']

    position = df.columns.get_loc(mobility_g)
    #position_2 = df.columns.get_loc(lockdown_g)
    #position_1 = df.columns.get_loc(normalcy_g)

    # Plot the dataframes on the same plot
    df.iloc[0, 3:].plot(ax=axs[i//2, i%2])
    df.iloc[1, 3:].plot(ax=axs[i//2, i%2])
    if country not in ['Korea', 'Serbia']:
      df.iloc[2, 3:].plot(ax=axs[i//2, i%2])
    df_mean.plot(ax=axs[i//2, i%2])

    axs[i//2, i%2].grid(True)
    axs[i//2, i%2].set_title('Mobility in ' + country)
    axs[i//2, i%2].set_xlabel('date')
    axs[i//2, i%2].set_ylabel('percentage of mobility compared to day 0')
    if country not in ['Korea', 'Serbia']:
      axs[i//2, i%2].legend(['Walking', 'Driving', 'Transit', 'Mean'])
    else:
      axs[i//2, i%2].legend(['Walking', 'Driving', 'Mean'])

    #axs[i//2, i%2].axvline(position_1-3, color='black', lw=2.2, linestyle="--")
    axs[i//2, i%2].axvline(position-3, color='blue', lw=1.5, linestyle="-", alpha=0.7)
    #axs[i//2, i%2].axvline(position_2-3, color='black', lw=1.5, linestyle="-", alpha=0.5)

  # Show the plot
  plt.tight_layout()
  plt.show()

def pageviewsplot(df_code, interventions, djson):
  fig, axs = plt.subplots(len(df_code)//2, 2, sharex=True, figsize=(20, 20))

  for i, c in enumerate(df_code['lang']):
      dt = djson[c]["topics"]["Culture.Media.Video games"]["percent"]

      mobility_g = interventions.loc[c]['Mobility']
      format_string = "%Y-%m-%d"

      # Convert the string to a numpy.datetime64 object
      date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

      dates = pd.to_datetime(list(dt.keys()))
      numbers = list(dt.values())

      if c == 'sv':
          x = [datetime.timestamp(k) for k in dates]
          x = x[365:]
          y = [val for val in numbers if not math.isnan(val)]
      else:
          x = [datetime.timestamp(k) for k in dates]
          y = numbers

      degree = 5
      coefficients = np.polyfit(x, y, degree)
      polynomial = np.poly1d(coefficients)

      y_fit = polynomial(x)

      row = i // 2
      col = i % 2

      axs[row, col].plot(dates, numbers)
      axs[row, col].plot(pd.to_datetime(x, unit='s'), y_fit)  # Convert x back to datetime for plotting

      axs[row, col].grid(True)

      axs[row, col].axvline(date_object, color='blue', lw=1.5, linestyle="-", alpha=0.7)
      axs[row, col].set_title('Percentage of Wikipedia page views related to video games in ' + df_code.index[i])
      axs[row, col].set_xlabel('Date')
      axs[row, col].set_ylabel('Percentage')
      axs[row, col].set_xlim(min(dates), max(dates))

      # Adjust x-axis labels
      # Get the dates for every 90 days
      selected_dates = pd.date_range(start=dates[0], end=dates[-1], freq='90D')

      # Format the dates as 'YYYY-MM-DD' and remove the time
      axs[row, col].set_xticks(selected_dates, selected_dates.strftime('%Y-%m-%d'), rotation=45)

  plt.tight_layout()
  plt.show()

def aggregatedpageviewsplot(df_code, djson, interventions):
  for i, c in enumerate(df_code['lang']):

      dt = djson[c]["topics"]["Culture.Media.Video games"]["percent"]
      dates = list(dt.keys())
      numbers = list(dt.values())
      dates = pd.to_datetime(dates)

      if c == 'sv':
          x = [datetime.timestamp(k) for k in dates]
          x = x[365:]
          y = [val for val in numbers if not math.isnan(val)]
      else:
          x = [datetime.timestamp(k) for k in dates]
          y = numbers

      #creating the approximated curve
      degree = 5
      coefficients = np.polyfit(x, y, degree)
      polynomial = np.poly1d(coefficients)
      y_fit = polynomial(x)

      #converting the mobility date (str) into a np.datetime64 to be able to use it
      mobility_g = interventions.loc[c]['Mobility']
      format_string = "%Y-%m-%d"
      date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

      index = dates.get_loc(date_object)
      mean = y_fit[0:index].mean()
      offset = 0 - mean
      y_fit = y_fit + offset

      plt.plot(pd.to_datetime(x, unit='s'), y_fit, label=c)  # Convert x back to datetime for plotting

      plt.grid(True)
      plt.title('Percentage of Wikipedia page views related to video games in depending on the country')
      plt.xlabel('Date')
      plt.ylabel('Percentage')
      plt.xlim(min(dates), max(dates))

      # Get the dates for every 90 days
      selected_dates = pd.date_range(start=dates[0], end=dates[-1], freq='90D')

      # Format the dates as 'YYYY-MM-DD' and remove the time
      plt.xticks(selected_dates, selected_dates.strftime('%Y-%m-%d'), rotation=45)

  plt.legend()
  plt.tight_layout()
  plt.show()

def desaggregatedpageviewsplot(df_code, djson, interventions):
  fig, axs = plt.subplots(1, 3, sharex = True, sharey = True, figsize=(18,6))

  for i, c in enumerate(df_code['lang']):

      dt = djson[c]["topics"]["Culture.Media.Video games"]["percent"]
      dates = list(dt.keys())
      numbers = list(dt.values())
      dates = pd.to_datetime(dates)

      if c == 'sv':
          x = [datetime.timestamp(k) for k in dates]
          x = x[365:]
          y = [val for val in numbers if not math.isnan(val)]
      else:
          x = [datetime.timestamp(k) for k in dates]
          y = numbers

      #creating the approximated curve
      degree = 5
      coefficients = np.polyfit(x, y, degree)
      polynomial = np.poly1d(coefficients)
      y_fit = polynomial(x)

      #converting the mobility date (str) into a np.datetime64 to be able to use it
      mobility_g = interventions.loc[c]['Mobility']
      format_string = "%Y-%m-%d"
      date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

      # Offset each curve
      index = dates.get_loc(date_object)
      mean = y_fit[0:index].mean()
      offset = 0 - mean
      y_fit = y_fit + offset

      col=0
      if c in ['fr', 'ca', 'it', 'sr']:
        row = 0
      else:
        if c in ['ko', 'ja', 'sv']:
          row = 2
        else:
          row = 1

      axs[row].plot(pd.to_datetime(x, unit='s'), y_fit, label=c)  # Convert x back to datetime for plotting

      axs[row].grid(True)
      axs[row].set_xlabel('Date')
      axs[row].set_ylabel('Percentage')
      axs[row].set_xlim(min(dates), max(dates))

      # Get the dates for every 90 days
      selected_dates = pd.date_range(start=dates[0], end=dates[-1], freq='90D')

      # Format the dates as 'YYYY-MM-DD' and remove the time
      axs[row].set_xticks(selected_dates, selected_dates.strftime('%Y-%m-%d'), rotation=45)

      axs[row].legend()

  axs[0].set_title('Variation of video games related page views \n for a very restrictive lockdown')
  axs[1].set_title('Variation of video games related page views \n for a restrictive lockdown')
  axs[2].set_title('Variation of video games related page views \n for an unrestrictive lockdown')
  plt.tight_layout()
  plt.show()

def correlationanalysisplot(df_code, globalmob, djson, interventions):

  fig, axs = plt.subplots(len(df_code)//2, 2, sharex=True, sharey = True, figsize=(20, 20))
  mean = meanmob(df_code, globalmob)

  for i, c in enumerate(df_code['lang']):

    av = mean[df_code.index[i]]

    if c == 'sv':
      zeros_360 = pd.Series([0] *360)
      zeros_25 = pd.Series([0] *25)
      mean_big = pd.concat([zeros_360, av, zeros_25])
      mean_big = mean_big.reset_index(drop=True)
      percent = pd.Series(djson[c]["topics"]["Culture.Media.Video games"]["percent"]).dropna()
    else:
      zeros_775 = pd.Series([0] *775)
      mean_big = pd.concat([zeros_775, av])
      mean_big = mean_big.reset_index(drop=True)
      mean_big = mean_big[:-25]
      percent = pd.Series(djson[c]["topics"]["Culture.Media.Video games"]["percent"])

    dates = list(percent.keys())
    dates = pd.to_datetime(dates)

    mobility_g = interventions.loc[c]['Mobility']
    format_string = "%Y-%m-%d"

    index = dates.get_loc(mobility_g)

    row = i // 2
    col = i % 2

    backwards = sm.tsa.ccf(percent, mean_big, adjusted=True)[::-1]
    forwards = sm.tsa.ccf(mean_big, percent, adjusted=True)
    ccf_output = np.r_[backwards[:-1], forwards]
    ccf_output = ccf_output[(len(ccf_output)//2)-50:(len(ccf_output)//2)+50]
    axs[row, col].stem(range(-len(ccf_output)//2, len(ccf_output)//2), ccf_output, markerfmt='.')

    # Fill the space between the curve and the zero line with color
    axs[row, col].fill_between(range(-len(ccf_output)//2, len(ccf_output)//2), ccf_output, 0, where=(ccf_output >= 0), interpolate=True, color='green', alpha=0.3, label='Positive Correlation')
    axs[row, col].fill_between(range(-len(ccf_output)//2, len(ccf_output)//2), ccf_output, 0, where=(ccf_output < 0), interpolate=True, color='red', alpha=0.3, label='Negative Correlation')

    axs[row, col].grid(True)
    axs[row, col].set_xlabel('Lag (Days)')
    axs[row, col].set_ylabel('Cross-Correlation')
    axs[row, col].set_title('Cross-Correlation between Lockdown Intensity and Video Game Page Views in ' + df_code.index[i])
    axs[row, col].legend()

  plt.tight_layout()
  plt.show()

def correlationanalysisplot_false(df_code, globalmob, djson):

  fig, axs = plt.subplots(len(df_code)//2, 2, sharex=True, figsize=(20, 20))
  mean = meanmob(df_code, globalmob)

  for i, c in enumerate(df_code['lang']):

    row = i // 2
    col = i % 2

    av = mean[df_code.index[i]]

    percent = pd.Series(djson[c]["topics"]["Culture.Media.Video games"]["percent"])
    p = {key[:10]: value for key, value in percent.items()}
    position = list(p.keys()).index('2020-02-15')

    backwards = sm.tsa.ccf(percent[position:], av[:-25], adjusted=True)[::-1]
    forwards = sm.tsa.ccf(av[:-25], percent[position:], adjusted=True)
    ccf_output = np.r_[backwards[:-1], forwards]
    ccf_output = ccf_output[(len(ccf_output)//2)-50:(len(ccf_output)//2)+50]
    axs[row, col].stem(range(-len(ccf_output)//2, len(ccf_output)//2), ccf_output, markerfmt='.')

    # Fill the space between the curve and the zero line with color
    axs[row, col].fill_between(range(-len(ccf_output)//2, len(ccf_output)//2), ccf_output, 0, where=(ccf_output >= 0), interpolate=True, color='green', alpha=0.3, label='Positive Correlation')
    axs[row, col].fill_between(range(-len(ccf_output)//2, len(ccf_output)//2), ccf_output, 0, where=(ccf_output < 0), interpolate=True, color='red', alpha=0.3, label='Negative Correlation')

    axs[row, col].grid(True)
    axs[row, col].set_xlabel('Lag (Days)')
    axs[row, col].set_ylabel('Cross-Correlation')
    axs[row, col].set_title('Cross-Correlation between Lockdown Intensity and Video Game Page Views in ' + df_code.index[i])
    axs[row, col].legend()

  plt.tight_layout()
  plt.show()

def pearsonpageviews_false(df_code, globalmob, djson):

  mean = meanmob(df_code, globalmob)
  for i, c in enumerate(df_code['lang']):

    country_code = df_code.iloc[i]['state']
    av = mean[df_code.index[i]]
    country,_ = defglobalmob(country_code, globalmob)

    percent = pd.Series(djson[c]["topics"]["Culture.Media.Video games"]["percent"])

    correlation_coefficient, p_value = pearsonr(av[:-25], percent[775:])
    print(f"For {c}, the correlation coefficient is: {correlation_coefficient}, and the p-value is: {p_value}")

def pearsonpageviews_plot(df_code, globalmob, djson):

  mean = meanmob(df_code, globalmob)

  # Lists to store correlation coefficients and corresponding p-values
  correlation_coefficients = []
  p_values = []

  for i, c in enumerate(df_code['lang']):
      country_code = df_code.iloc[i]['state']
      av = mean[df_code.index[i]]
      country, _ = defglobalmob(country_code, globalmob)

      percent = pd.Series(djson[c]["topics"]["Culture.Media.Video games"]["percent"])

      # Calculate correlation coefficient and p-value
      correlation_coefficient, p_value = pearsonr(av[:-25], percent[775:])

      correlation_coefficients.append(correlation_coefficient)
      p_values.append(p_value)

  # Create a DataFrame for easy plotting with Seaborn
  df_plot = pd.DataFrame({'Country': df_code.index, 'Correlation Coefficient': correlation_coefficients, 'P-value': p_values})

  # Set the style of seaborn
  sns.set(style="whitegrid")

  # Create a bar plot
  plt.figure(figsize=(10, 6))
  ax = sns.barplot(x='Country', y='Correlation Coefficient', data=df_plot, palette=['red' if p > 0.05 else 'green' for p in p_values])
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

  # Add labels and title
  plt.xlabel('Country')
  plt.ylabel('Correlation Coefficient')
  plt.title('Correlation Coefficients between the moblity in the country and the pageviews')
  legend_handles = [
    Line2D([0], [0], marker='s', color='w', label='p-value > 0.05', markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='p-value < 0.05', markerfacecolor='green', markersize=10)
    ]
  plt.legend(handles=legend_handles, loc='upper left')

  plt.show()

def pearsonpageviews_plot_inter(df_code, globalmob, djson):
    mean = meanmob(df_code, globalmob)

    correlation_coefficients = []
    p_values = []

    for i, c in enumerate(df_code['lang']):
        country_code = df_code.iloc[i]['state']
        av = mean[df_code.index[i]]
        country, _ = defglobalmob(country_code, globalmob)

        percent = pd.Series(djson[c]["topics"]["Culture.Media.Video games"]["percent"])

        correlation_coefficient, p_value = pearsonr(av[:-25], percent[775:])

        correlation_coefficients.append(correlation_coefficient)
        p_values.append(p_value)

    df_plot = pd.DataFrame({'Country': df_code.index, 'Correlation Coefficient': correlation_coefficients, 'P-value': p_values})

    # Create traces for each bar
    traces = []
    for i, country in enumerate(df_plot['Country']):

        if country in ['France', 'Catalonia', 'Italy', 'Serbia']:
          g = "groupe very restrictive"
        else:
          if country in ['Korea', 'Japan', 'Sweden']:
            g = "group unrestrictive"
          else:
            g = "group restrictive"

        trace = go.Bar(
            x=[country],
            y=[df_plot.loc[i, 'Correlation Coefficient']],
            name=country,
            marker=dict(color='red' if df_plot.loc[i, 'P-value'] > 0.05 else 'green'),
            legendgroup=g,
            legendgrouptitle_text=g,
            visible=True  # All traces are initially visible
        )
        traces.append(trace)

    layout = go.Layout(
        title='Correlation Coefficients between the mobility in the country and the pageviews',
        xaxis=dict(title='Country', range=[min(df_plot['Country']), max(df_plot['Country'])]),  # Set range for x-axis
        yaxis=dict(title='Correlation Coefficient', range=[min(df_plot['Correlation Coefficient'])-0.05, max(df_plot['Correlation Coefficient'])+0.05]),  # Set range for y-axis
        showlegend=True,
    )

    # Create frames for animation
    frames = [go.Frame(data=[trace], name=country) for country, trace in zip(df_plot['Country'], traces)]
    fig = go.Figure(data=traces, layout=layout, frames=frames)

    fig.show()

def medianpercentage_plot(df_code, globalmob, djson, interventions):

  data1 = {'Country': [],
          'percent': []}

  df_perc = pd.DataFrame(data1)

  for i, c in enumerate(df_code['lang']):

      cs = df_code.index[i]
      df, mobility = defglobalmob(cs, globalmob)
      dt = djson[c]["topics"]["Culture.Media.Video games"]["percent"]

      mobility_g = interventions.loc[c]['Mobility']
      normalcy_g = interventions.loc[c]['Normalcy']
      format_string = "%Y-%m-%d"

      # Convert the string to a numpy.datetime64 object
      date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

      dates = list(dt.keys())
      numbers = list(dt.values())

      dates = pd.to_datetime(dates)

      if c == 'sv':
          x = [datetime.timestamp(k) for k in dates]
          x = x[365:]
          y = [val for val in numbers if not math.isnan(val)]
      else:
          x = [datetime.timestamp(k) for k in dates]
          y = numbers

      df['date'] = pd.to_datetime(df['date'], utc=None)
      p = {key[:10]: value for key, value in dt.items()}
      position = list(p.keys()).index('2020-02-15')
      position2 = list(p.keys()).index(mobility_g)
      position3 = list(p.keys()).index(normalcy_g)

      median1 = np.nanmedian(numbers[position-30:position2])
      median2 =  np.nanmedian(numbers[position2:position3])

      percentage = (median2-median1)*100/median1

      new_row = pd.DataFrame({'Country': [cs], 'percent': [percentage], 'median_after_lockdown': [median2]})
      df_perc = pd.concat([df_perc, new_row], ignore_index=True)

  def get_color(country_code):
      if country_code in ['France', 'Catalonia', 'Italy', 'Serbia']:
          return 'tab:red'
      elif country_code in ['Korea', 'Japan', 'Sweden']:
          return 'yellow'
      else:
          return 'orange'

  # Apply the color function to create a 'color' column in the DataFrame
  df_perc['color'] = df_perc['Country'].apply(get_color)

  plt.figure(figsize=(10, 7))
  ax = sns.barplot(x='Country', y='percent', data=df_perc, palette=df_perc['color'])

  # Setting grid lines, title, and labels
  plt.grid(True)
  plt.title('Change in the proportions of pageviews related to \n video games amongst all the pageviews (before and after lockdown)')
  plt.xlabel('Countries')
  plt.ylabel('Percentage change')
  plt.xticks(rotation=45)
  plt.axhline(0, color='black', lw=1.5, linestyle="-", alpha=0.7)

  legend_handles = [
  Line2D([0], [0], marker='s', color='w', label='countries with very restrictive lockdown', markerfacecolor='tab:red', markersize=10),
  Line2D([0], [0], marker='s', color='w', label='countries with restrictive lockdown', markerfacecolor='orange', markersize=10),
  Line2D([0], [0], marker='s', color='w', label='countries with unrestrictive lockdown', markerfacecolor='yellow', markersize=10),
  ]
  plt.legend(handles=legend_handles, loc='upper right')

  # Show the plot
  plt.show()

def medianpercentage_boxplot(df_code, globalmob, djson, interventions):

  data1 = {'Country': [],
        'percent': [],
        'color': []}

  for i, c in enumerate(df_code['lang']):

      cs = df_code.index[i]
      df, mobility = defglobalmob(cs, globalmob)
      dt = djson[c]["topics"]["Culture.Media.Video games"]["percent"]

      mobility_g = interventions.loc[c]['Mobility']
      normalcy_g = interventions.loc[c]['Normalcy']
      format_string = "%Y-%m-%d"

      # Convert the string to a numpy.datetime64 object
      date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

      dates = list(dt.keys())
      numbers = list(dt.values())

      dates = pd.to_datetime(dates)

      if c == 'sv':
          x = [datetime.timestamp(k) for k in dates]
          x = x[365:]
          y = [val for val in numbers if not math.isnan(val)]
      else:
          x = [datetime.timestamp(k) for k in dates]
          y = numbers

      df['date'] = pd.to_datetime(df['date'], utc=None)
      p = {key[:10]: value for key, value in dt.items()}
      position = list(p.keys()).index('2020-02-15')
      position2 = list(p.keys()).index(mobility_g)
      position3 = list(p.keys()).index(normalcy_g)

      median1 = np.nanmedian(numbers[position-30:position2])
      median2 =  np.nanmedian(numbers[position2:position3])

      percentage = (median2-median1)*100/median1

      def get_color(country_code):
        if country_code in ['France', 'Catalonia', 'Italy', 'Serbia']:
            return 'very restrictive'
        elif country_code in ['Korea', 'Japan', 'Sweden']:
            return 'unrestricitive'
        else:
            return 'restrictive'

      color = get_color(cs)

      data1['Country'].append(cs)
      data1['percent'].append(percentage)
      data1['color'].append(color)

  # Create DataFrame
  df_perc = pd.DataFrame(data1)

  fig = px.box(df_perc, x='color', y='percent', color = 'color', points='all', notched = True,
                  labels={'Correlation Coefficient': 'Correlation Coefficient'},
                  title='Correlation Coefficients between the mobility in the country and the pageviews')

  # Setting layout options
  fig.update_layout(
      title='Change in the proportions of pageviews related to video games (before and after lockdown)',
      xaxis=dict(title='Countries'),
      yaxis=dict(title='Percentage Change'),
      showlegend=True,
      legend=dict(title='Lockdown Category')
  )

  # Show the plot
  fig.show()

  # Output html that you can copy paste
  fig.to_html(full_html=False, include_plotlyjs='cdn')
  # Saves a html doc that you can copy paste
  fig.write_html("boxplot.html", full_html=False, include_plotlyjs='cdn')

# Returns the df_query for the given url
def return_wiki_fetched(url):
    # Code for the language of the countries for the following study
    country_code = ['en', 'fr', 'it', 'de', 'ja']

    headers = {'User-Agent':'ADABot/0.0 (floydchow7@gmail.com)'}

    try:
      r = requests.get(url, headers=headers)
      df_onequery = pd.DataFrame(r.json()['items'])
      time.sleep(0.5) # In case the IP address is blocked
      return df_onequery
    except:
      print('The {} page views are not found during these time'.format(country))

def fetch_wikiviews():

    df_wikiviews = pd.DataFrame()
    
    country_code = ['en', 'fr', 'it', 'de', 'ja']

    for country in country_code:
        # Retrieve page views for the entire wikipedia for a particular country:
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/{country}.wikipedia/all-access/user/daily/2017010100/2023120100"
        df_wikiviews = pd.concat([df_wikiviews, return_wiki_fetched(url)])

    # Additional retrieval for all languages to show the change of views during lockdown
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/all-projects/all-access/user/daily/2017010100/2023120100"
    df_wikiviews = pd.concat([df_wikiviews, return_wiki_fetched(url)])

    # Data Wrangling:
    df_wikiviews = df_wikiviews[['project', 'timestamp', 'views']].reset_index(drop=True)
    df_wikiviews['timestamp'] = pd.to_datetime(df_wikiviews['timestamp'], format='%Y%m%d%H')
    df_wikiviews['project'] = df_wikiviews['project'].str.replace(r'\..*', '', regex=True)
    df_wikiviews = df_wikiviews.pivot_table(index = 'timestamp', columns = ['project'], values = 'views')

    return df_wikiviews

def get_monthly(df_wikiviews, df_interventions):

    # Plot the total views on wikipedia to show the impact of lockdown
    df_lang = df_interventions[df_interventions['lang'] == 'en']

    # Extract the monthly overall views for the bar plot
    df_wikiviews['Month'] = df_wikiviews.index.to_period('M')
    monthly_data = df_wikiviews.groupby('Month')['all-projects'].sum()
    df_wikiviews.drop(['Month'], axis=1)
    monthly_data.index = monthly_data.index.to_timestamp()

    return monthly_data

def mehdi_p1(df_wikiviews, df_interventions):

    monthly_data = get_monthly(df_wikiviews, df_interventions)

    fig, ax = plt.subplots(figsize=(20, 5))

    bars = plt.bar(monthly_data.index[:-1], monthly_data[:-1], label='Entire Wikipedia Pageviews', color='blue', alpha=0.7, width=20)

    # Change the color of lockdown months
    for date in ['2020-03-01', '2020-04-01', '2020-05-01']:
        idx = np.where(monthly_data.index == pd.to_datetime(date))[0]
        idx = idx[0]
        bars[idx].set_color('red')

    bars[idx].set_label('Lockdown Period')
    plt.legend()
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Page Views', fontsize=12)
    plt.title('Page Views Over Time per Month', fontsize=14)
    plt.ylim(1.3e10, 2.2*1e10)
    plt.xticks(rotation=45)

    plt.show()

    return 

def mehdi_2(df_wikiviews, df_interventions):

    monthly_data = get_monthly(df_wikiviews, df_interventions)
    # Months to test for significant changes
    target_months = ['2020-03-01', '2020-04-01', '2020-05-01']

    # Baseline months (all before covid)
    baseline_months = list(monthly_data.loc[:'2020-02-01'].index)

    target_data = monthly_data[monthly_data.index.isin(target_months)]
    baseline_data = monthly_data[monthly_data.index.isin(baseline_months)]

    t_values, p_values = [], []

    for month in target_months:
        target_views = target_data.loc[target_data.index == month]
        baseline_views = baseline_data
        t_stat, p_value = stats.ttest_ind(baseline_views, target_views, equal_var=False)
        t_values.append(t_stat)
        p_values.append(p_value)
        # print(f"Month: {month}")
        # print(f"t-statistic: {t_stat}")
        print(f"p-value: {p_value}")
        print("Significant change" if p_value < 0.05 else "No significant change")
        print()

    # Bar Plot to show if the changes are significant for the months studied
    significance_level = 0.05

    plt.figure(figsize=(10, 6))

    plt.bar(target_months, p_values, color='blue', alpha=0.7, label='t-values')
    plt.axhline(y=significance_level, color='red', linestyle='--', label='Critical t-value (Significance level = 0.05)')

    for i in range(len(target_months)):
        plt.text(target_months[i], p_values[i], f'p={p_values[i]:.4f}', ha='center', va='bottom')

    plt.xlabel('Months')
    plt.ylabel('p-value')
    plt.title('Comparison of Monthly Data with Baseline (t-test Results)')
    plt.legend()

    plt.show()

def fetch_chess():

    list_chess = ["Chess", "Makruk", "Chaturanga", "Janggi", "Xiangqi", "Sittuyin", "Shogi"]
    country_code = ['en', 'fr', 'it', 'de', 'ja']

    df_chess = pd.DataFrame()

    for name in list_chess:
    # Retrieve page views for the entire wikipedia for a particular country:
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{name}/daily/2017010100/2023120100"
        df_chess = pd.concat([df_chess, return_wiki_fetched(url)])

    # Data Wrangling:
    df_chess = df_chess[['project', 'article', 'timestamp','views']].reset_index(drop=True)
    df_chess['timestamp'] = pd.to_datetime(df_chess['timestamp'], format='%Y%m%d%H')
    df_chess['project'] = df_chess['project'].str.replace(r'\..*', '', regex=True)
    df_chess = df_chess.pivot_table(index = 'timestamp', columns = ['article'], values = 'views')
    df_chess.columns.set_names(['Game Name'], inplace=True)

    return df_chess

def return_figure(column_name, df, ax, title, df_interventions, language='en', weekends=False):

    # Extract intervention dates for specific languages
    df_lang = df_interventions[df_interventions['lang']==language]

    # Plot taking into acount that we can have a pd.Series as entry
    if isinstance(df, pd.Series):
      ax.plot(df.index, df, label=language)
    else:
      ax.plot(df.index, df[language], label=language)

    # Plots a span for the period of lockdown in red
    lockdown_start = df_lang['Mobility'].iloc[0]
    lockdown_end = df_lang['Normalcy'].iloc[0]
    ax.axvspan(lockdown_start, lockdown_end, color='red', alpha=0.3, label='Lockdown Period')

    # Plot grey areas on weekends
    if weekends:
        weekends = [d for d in df.index if d.weekday() in [5, 6]]  # 5 = Saturday, 6 = Sunday
        for start in weekends:
            # Check if the next day is also a weekend
            if start + pd.Timedelta(days=1) in weekends:
                end = start + pd.Timedelta(days=1)
                ax.axvspan(start, end, facecolor='gray', edgecolor=None, alpha=0.2)

    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel('Date')
    ax.set_ylabel('Views on the page (log scale)')
    ax.set_title(title)

    return ax

def interactive_chess(df_chess, df_interventions):

    game_selector = widgets.Dropdown(
        options=["Chess", "Makruk", "Chaturanga", "Janggi", "Xiangqi", "Sittuyin", "Shogi"],
        description="Select a Game:",
        disabled=False
    )

    # Interactive plot to show selected game for english language
    @interact(game_name=game_selector)
    def update_plot(game_name):
        fig, ax = plt.subplots(figsize=(32, 10))
        ax = return_figure(game_name, df_chess[game_name], ax, f'Page Views Over Time for {game_name} in English', df_interventions, weekends=True)
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        plt.show()

# Use the API to find the name of the board Games in different languages
def get_langlinks(list_names):
    
    wiki_wiki = wikipediaapi.Wikipedia('ADABot/0.0 (floydchow7@gmail.com)', 'en')
    
    list_names_languages = {}
    
    for name_original in list_names:
        name = name_original.replace(' ', '_')
        page = wiki_wiki.page(name)
        langlinks = page.langlinks
        list_lang = {'en': name_original, 'fr': None, 'it': None, 'de': None, 'ja': None}
        for k in sorted(langlinks.keys()):
            v = langlinks[k]
            if v.language in list(list_lang.keys()):
                list_lang[v.language] = v.title
        # Save the names in a dictionnary to extract them later
        list_names_languages[name] = list_lang
    return list_names_languages

def fetch_boardgames():
    # Name of the board games in English
    list_boardgames_original = {
        "Catan",
        "Ticket to Ride (board game)",
        "Pandemic (board game)",
        "Carcassonne (board game)",
        "Risk (game)",
        "Monopoly (game)",
        "Scrabble",
        "Chess",
        "Backgammon",
        "Cluedo",
        "Twilight Struggle",
        "Agricola (board game)",
        "7 Wonders (board game)",
        "Terraforming Mars (board game)",
        "Dominion (card game)",
        "Power Grid",
        "Betrayal at House on the Hill",
        "Splendor (game)",
        "Gloomhaven",
        "Azul (board game)",
        "Chess"
    }
    
    country_code = ['en', 'fr', 'it', 'de', 'ja']

    list_boardgames = get_langlinks(list_boardgames_original)

    df_boardgames = pd.DataFrame()

    for name in list_boardgames:
        for language in country_code:
            name_lang = list_boardgames[name][language]
            if name_lang != None:
                # Retrieve page views for the entire wikipedia for a particular country:
                url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{language}.wikipedia/all-access/user/{name_lang}/daily/2017010100/2023120100"
                df_onequery = return_wiki_fetched(url)
                df_onequery['article'] = name
                df_boardgames = pd.concat([df_boardgames,df_onequery])
    
    # Data wrangling:
    df_boardgames = df_boardgames[['project', 'article', 'timestamp','views']].reset_index(drop=True)
    df_boardgames['timestamp'] = pd.to_datetime(df_boardgames['timestamp'], format='%Y%m%d%H')
    df_boardgames['project'] = df_boardgames['project'].str.replace(r'\..*', '', regex=True)
    df_boardgames = df_boardgames.pivot_table(index = 'timestamp', columns = ['article', 'project'], values = 'views')
    df_boardgames.columns.set_names(['Game Name', 'Language'], level=[0, 1], inplace=True)

    return df_boardgames, list_boardgames

def interactive_boardgames(df_boardgames, list_boardgames, df_interventions):

    game_selector = widgets.Dropdown(
        options=list(list_boardgames.keys()),
        description="Select a Game:",
        disabled=False,
    )

    language_selector = widgets.Dropdown(
        options=['en', 'fr', 'it', 'de', 'ja'],
        description="Select a Language Option:",
        disabled=False,
    )

    # Interactive plot to show selected game for selected language
    @interact(game_name=game_selector, language_name=language_selector)
    def update_plot(game_name, language_name):
        clear_output(wait=True)  # Clear previous output
        fig, ax = plt.subplots(figsize=(32, 10))
        try:
            ax = return_figure(game_name, df_boardgames[game_name], ax, f'Page Views Over Time for {game_name}', df_interventions, language_name)
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            fig.autofmt_xdate(rotation=45)  # Rotates the x-axis labels for better readability
            fig.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
            plt.show()
        except:
            print('The pageviews is not disponible for this game in this language, please choose another combination.')

def extract_bg_en(df_boardgames):

    # Extract the pageviews for boardgames in English language
    df_boardgames_en = pd.DataFrame()

    for column in df_boardgames.columns:
        if column[1] == 'en':
            df_boardgames_en[column[0]] = df_boardgames[column]
    
    return df_boardgames_en

def return_fig_boardgames_lang(fig, ax1, language, df_boardgames, df_interventions, df_wikiviews):

    # Extract the data of a particular board game
    df = pd.DataFrame()
    for column in df_boardgames.columns:
        if column[1] == language:
            df[column[0]] = df_boardgames[column]

    ax1.plot(df.index, df.sum(axis=1).rolling(window=7).mean(), label = 'Interest for Board Games', alpha=0.7, color = 'blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Y-axis (Board Games Views)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis on the right
    ax2 = ax1.twinx()

    ax2.plot(df_wikiviews.index, df_wikiviews[language].rolling(window=7).mean(), label = f'Interest for Wikipedia {language}', color = 'red', alpha=0.8)
    ax2.set_ylabel(f'Y-axis ({language} Wikipedia Views)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Adds a red span conresponding to the lockdown period
    df_lang = df_interventions[df_interventions['lang']==language]
    lockdown_start = df_lang['Mobility'].iloc[0]
    lockdown_end = df_lang['Normalcy'].iloc[0]
    plt.axvspan(lockdown_start, lockdown_end, color='red', alpha=0.3, label='Lockdown Period')

    plt.title(f'Pageviews for Board Games and Wikipedia for {language} (smoothened)')

    return fig

def interactive_boardgames_lang(df_boardgames, df_interventions, df_wikiviews):
    language_selector = widgets.Dropdown(
        options=['en', 'fr', 'it', 'de', 'ja'],
        description="Select a Language Option:",
        disabled=False,
    )

    # Creates interactive plot and changes it given the language
    @interact(language=language_selector)
    def update_plot(language):
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(16, 5))
        fig = return_fig_boardgames_lang(fig, ax, language, df_boardgames, df_interventions, df_wikiviews)
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        plt.show()

def show_standardized_boardgames_en(df_interventions, df_boardgames_en, df_wikiviews):
    scaler_standardization = StandardScaler()

    # Extracting language dates for English
    df_lang = df_interventions[df_interventions['lang']=='en']

    lockdown_start = df_lang['Mobility'].iloc[0]
    lockdown_end = df_lang['Normalcy'].iloc[0]

    # Standardization of input data and actual data to compare the results
    stand_wiki = scaler_standardization.fit_transform(df_wikiviews[['en']])
    temp = pd.DataFrame({'temp': df_boardgames_en.sum(axis=1)})
    stand_boardgames = scaler_standardization.fit_transform(temp[['temp']])

    # Smoothened data to ease the reading
    df_data = pd.DataFrame({'wiki': stand_wiki[:, 0], 'board': stand_boardgames[:, 0]})
    window_size = 7
    smoothed_data1 = df_data['wiki'].rolling(window=window_size).mean()
    smoothed_data2 = df_data['board'].rolling(window=window_size).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df_boardgames_en.index, smoothed_data2, label = 'Standardized Board Games Page Views', alpha=0.7, color = 'blue')
    plt.plot(df_wikiviews.index, smoothed_data1, label = 'Standardized Wikipedia Page Views', color = 'red', alpha=0.8)
    plt.axvspan(lockdown_start, lockdown_end, color='red', alpha=0.3, label='Lockdown Period')
    plt.xlabel('Date')
    plt.ylabel('Standardized Number of Views')
    plt.title('Standardized PageViews in Board Games and Wikipedia for English Language (smoothened)')
    plt.legend()
    plt.grid()
    plt.show()

def pre_process_data(df_boardgames_en, df_wikiviews):
    # Data pre processing for the model:

    data = pd.DataFrame({'interest': df_boardgames_en.sum(axis=1)})
    # Data to train the model taken before 2020 to properly show the influence of the pandemic
    data = data[(data.index <= pd.to_datetime(datetime(2020,1,1)))]

    data['interest_wikipedia'] = df_wikiviews['en']

    data = data.dropna()

    # Standardization to better the meaningfulness of the results
    scaler_standardization = StandardScaler()
    data['interest'] = scaler_standardization.fit_transform(data[['interest']])
    data['interest_wikipedia'] = scaler_standardization.fit_transform(data[['interest_wikipedia']])

    # Defining feature and target variables
    X = data['interest_wikipedia']
    y = data['interest']

    # Splitting the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Creating sequences of 20 days to use for the prediction
    sequences = []

    for i in range(10, len(X)-9):
        sequences.append([X[i-10], X[i-9], X[i-8], X[i-7], X[i-6], X[i-5], X[i-4],
                        X[i-3], X[i-2], X[i-1], X[i], X[i+1], X[i+2], X[i+3],
                        X[i+4], X[i+5], X[i+6], X[i+7], X[i+8], X[i+9]])

    sequence_length = len(sequences[0])
    feature_dim = len(sequences[0])

    # Creating a tensor for the input data (sequences)
    input_tensor = torch.zeros(len(sequences), sequence_length, feature_dim, dtype=torch.float32)

    for i, sequence in enumerate(sequences):
        input_tensor[i, :, :] = torch.tensor(sequence, dtype=torch.float32)
    
    return (input_tensor, y)

# Defining the LSTM-based neural network model
class LSTMRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMRegression, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        out = self.fc(lstm_out[:, -1, :])
        return out

def train_evaluate_LSTM_model(input_tensor, y):
    # Defining the model's hyperparameters
    input_dim = input_tensor.shape[2]
    hidden_dim = 64
    num_layers = 2
    output_dim = 1

    # Initializing the model
    model = LSTMRegression(input_dim, hidden_dim, num_layers, output_dim)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Splitting the data
    X_train_seq, X_test_seq, y_train, y_test = train_test_split(input_tensor, y[10:-9], test_size=0.5, random_state=42)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Training of the model
    epochs = 50
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_seq)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Make predictions on the test set
    with torch.no_grad():
        y_pred_tensor = model(X_test_seq).numpy()

    y_pred = y_pred_tensor.flatten()

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    return model

def test_model(df_boardgames_en, df_interventions, df_wikiviews, model):

    scaler_standardization = StandardScaler()

    # Standardizing the pageviews of English Wikipedia of Board Games
    normalized = scaler_standardization.fit_transform(df_boardgames_en.sum(axis=1).values.reshape(-1, 1))
    normalized = normalized[:, 0]

    # Standardizing the pageviews of English Wikipedia to predict the behavior of Board Games Pageviews
    normalized_wiki = scaler_standardization.fit_transform(df_wikiviews['en'].values.reshape(-1, 1))
    normalized_wiki = normalized_wiki[:, 0]

    new_data = []

    for i in range(10, len(normalized_wiki)-9):
        new_data.append([normalized_wiki[i-10], normalized_wiki[i-9], normalized_wiki[i-8], normalized_wiki[i-7], normalized_wiki[i-6], normalized_wiki[i-5], normalized_wiki[i-4],
                        normalized_wiki[i-3], normalized_wiki[i-2], normalized_wiki[i-1], normalized_wiki[i], normalized_wiki[i+1], normalized_wiki[i+2], normalized_wiki[i+3],
                        normalized_wiki[i+4], normalized_wiki[i+5], normalized_wiki[i+6], normalized_wiki[i+7], normalized_wiki[i+8], normalized_wiki[i+9]])
    
    # Create an empty tensor to hold the input data
    sequence_length = len(new_data[0])
    feature_dim = len(new_data[0])
    input_tensor = torch.zeros(len(new_data), sequence_length, feature_dim, dtype=torch.float32)

    for i, new_data in enumerate(new_data):
        input_tensor[i, :, :] = torch.tensor(new_data, dtype=torch.float32)

    # Predicting the standardized Board Games Pageviews
    with torch.no_grad():
        predicted = model(input_tensor).numpy()
    predicted = predicted.flatten()

    # Get the dates for the lockdown:
    # Extracting language dates for English
    df_lang = df_interventions[df_interventions['lang']=='en']

    lockdown_start = df_lang['Mobility'].iloc[0]
    lockdown_end = df_lang['Normalcy'].iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(df_boardgames_en.index, normalized, label = 'Actual Level', alpha=0.7, color = 'blue')
    plt.plot(df_boardgames_en.index[10:-9], predicted, label = 'Predicted Level based on Wikipedia Page Views', color = 'red', alpha=0.8)
    plt.axvspan(lockdown_start, lockdown_end, color='red', alpha=0.3, label='Lockdown Period')
    plt.xlabel('Date')
    plt.ylabel('Standardized Page Views')
    plt.title('Actual vs. Predicted Interest in Board Games')
    plt.legend()
    plt.grid()
    plt.show()

    # Smoothening the plot for a window of a week
    df1 = pd.DataFrame({'pred': predicted})
    df2 = pd.DataFrame({'actual': normalized})

    window_size = 7
    smoothed_data1 = df1['pred'].rolling(window=window_size).mean()
    smoothed_data2 = df2['actual'].rolling(window=window_size).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df_boardgames_en.index, smoothed_data2, label = 'Actual Interest', alpha=0.7, color = 'blue')
    plt.plot(df_boardgames_en.index[10:-9], smoothed_data1, label = 'Predicted Interest based on Wikipedia Page Views', color = 'red', alpha=0.8)
    plt.axvspan(lockdown_start, lockdown_end, color='red', alpha=0.3, label='Lockdown Period')
    plt.xlabel('Date')
    plt.ylabel('Standardized PageViews')
    plt.title('Actual vs. Predicted Interest in Board Games')
    plt.legend()
    plt.grid()
    plt.show()

def get_metrics():
    metrics = [
        ('Difference_Strategy', 'Ratio_Strategy'),
        ('Difference_Action', 'Ratio_Action'),
        ('Difference_Adult', 'Ratio_Adult'),
        ('Difference_Miscellaneous', 'Ratio_Miscellaneous')
    ]
    return metrics

def prepare_dataframe_for_timeseries(df, timestamp_column='timestamp', date_format='%Y-%m-%d'):
    """
    Prepares a DataFrame for time series analysis by converting a specified timestamp column 
    to datetime, setting it as the DataFrame index, and ensuring the index is in 
    the correct datetime format.

    Parameters:
    df (pd.DataFrame): DataFrame to be processed.
    timestamp_column (str): Name of the column containing the timestamp.
    date_format (str): Format of the timestamp in the original DataFrame.

    Returns:
    pd.DataFrame: Processed DataFrame ready for time series analysis.
    """
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], format=date_format)
    df.set_index(timestamp_column, inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def plot_comparison_subplots(comparison_final, metrics):
   
    num_rows = len(metrics)
    
    # Create a list of subplot titles based on the metrics provided.
    # Ensure the titles correspond to the correct 'Difference' or 'Ratio' metric.
    subplot_titles = []
    for metric_difference, metric_ratio in metrics:
        subplot_titles.append(metric_difference.replace('_', ' '))
        subplot_titles.append(metric_ratio.replace('Ratio_', 'Percentage '))
    
    # Create a subplot grid with a specified number of rows and 2 columns
    # and include the subplot titles.
    fig = make_subplots(rows=num_rows, cols=2, shared_xaxes=True,
                        vertical_spacing=0.1,  # Increase spacing to accommodate titles
                        subplot_titles=subplot_titles)

    # Add traces for difference and ratio in their respective columns
    for row, (metric_difference, metric_ratio) in enumerate(metrics, start=1):
        # Add a bar plot for the metric difference on the left column
        fig.add_trace(
        go.Bar(
            x=comparison_final['country'],
            y=comparison_final[metric_difference],
            name=metric_difference,  # Set this to the metric's name
            marker_color='green'
        ),
        row=row, col=1
    )
    
        # Add a bar plot for the metric ratio on the right column
        fig.add_trace(
        go.Bar(
            x=comparison_final['country'],
            y=comparison_final[metric_ratio],
            name=metric_ratio,  # Set this to the metric's name
            marker_color='grey'
        ),
        row=row, col=2
    )

        # Update y-axis titles
        fig.update_yaxes(title_text='Difference in Means', row=row, col=1)
        fig.update_yaxes(title_text='% of Change(%)', row=row, col=2)
        
    # Update x-axis titles for the last row only
    fig.update_xaxes(title_text='Country', row=num_rows, col=1)
    fig.update_xaxes(title_text='Country', row=num_rows, col=2)

    # Adjust the layout and show the plot
    fig.update_layout(
        height=200 * num_rows,  # Adjust height to fit titles
        width=800,
        title_text='Comparison of different topics by Country',
        showlegend=False
    )
    fig.show()
    fig.write_html("navie.html")

def get_causal_impact_data():
    
    causal_impact_data = {
        'Action': {
            'France_Before': 41.25, 'France_After': 31.21, 'France_Sig': 1,
            'Italy_Before': 54.85, 'Italy_After': 23.33, 'Italy_Sig': 1,
            'Japan_Before': 14.68, 'Japan_After': -3.67, 'Japan_Sig': 0,
            'Germany_Before': -7.95, 'Germany_After': -9.22, 'Germany_Sig': 0
        },
        'Adult': {
            'France_Before': 53.65, 'France_After': 43.93, 'France_Sig': 1,
            'Italy_Before': 41.68, 'Italy_After': 13.39, 'Italy_Sig': 1,
            'Japan_Before': 34.28, 'Japan_After': 15.26, 'Japan_Sig': 1,
            'Germany_Before': 17.62, 'Germany_After': 12.72, 'Germany_Sig': 0
        },
        'Strategy': {
            'France_Before': 26.64, 'France_After': 15.23, 'France_Sig': 1,
            'Italy_Before': 40.58, 'Italy_After': 10.51, 'Italy_Sig': 1,
            'Japan_Before': 45.28, 'Japan_After': 21.43, 'Japan_Sig': 1,
            'Germany_Before': 27.41, 'Germany_After': 23.76, 'Germany_Sig': 1
        },
        'Miscellaneous': {
            'France_Before': 43.46, 'France_After': 33.53, 'France_Sig': 1,
            'Italy_Before': 64.94, 'Italy_After': 29.97, 'Italy_Sig': 1,
            'Japan_Before': 12.64, 'Japan_After': -6.12, 'Japan_Sig': 0,
            'Germany_Before': 129.32, 'Germany_After': 124.63, 'Germany_Sig': 1
        }
    }
    return causal_impact_data

def plot_causal_impact_with_updated_legend(causal_impact_data):
    
    # Create subplots: 2 rows, 2 columns
    fig = make_subplots(rows=2, cols=2, subplot_titles=tuple(causal_impact_data.keys()))

    # Define subplot position mapping
    subplot_pos = [(1, 1), (1, 2), (2, 1), (2, 2)]

    # Create a bar plot for each category
    for i, (category, pos) in enumerate(zip(causal_impact_data, subplot_pos)):
        data = causal_impact_data[category]
        countries = ['France', 'Italy', 'Japan', 'Germany']
        before_values = [data[country + '_Before'] for country in countries]
        after_values = [data[country + '_After'] for country in countries]
        significance = [data[country + '_Sig'] for country in countries]

        # Creating bar colors based on significance
        colors_before = ['green' if sig else 'darkgrey' for sig in significance]
        colors_after = ['darkgreen' if sig else 'grey' for sig in significance]

        # Plotting the 'Before' data
        fig.add_trace(go.Bar(x=countries, y=before_values, name='Before Division With Significance', 
                             marker_color=colors_before, legendgroup='Before', showlegend=(i==0)), 
                      row=pos[0], col=pos[1])

        # Plotting the 'After' data
        fig.add_trace(go.Bar(x=countries, y=after_values, name='After Division With Significance', 
                             marker_color=colors_after, legendgroup='After', showlegend=(i==0)), 
                      row=pos[0], col=pos[1])

    # Update layout for the figure
    fig.update_layout(
        title_text="Causal Impact of COVID-19 on Game Categories（Grey plot means without significance）",
        height=500,
        showlegend=True,
        legend=dict(
            itemsizing='constant',
            traceorder='grouped',
            orientation='h',
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update xaxis and yaxis titles
    for i in range(1, 5):
        fig['layout']['xaxis' + str(i)].title.text = 'Countries'
        fig['layout']['yaxis' + str(i)].title.text = 'Relative Impact (%)'

    # Show the plot
    fig.show()
    fig.write_html("casual.html")

def merge_and_group_data(pageviews, game_genres):
    """
    Renames columns in the pageviews DataFrame, merges it with the game_genres DataFrame, 
    and groups and aggregates the views.

    Parameters:
    pageviews (pd.DataFrame): DataFrame containing pageviews data.
    game_genres (pd.DataFrame): DataFrame containing game genres data.

    Returns:
    pd.DataFrame: A grouped and aggregated DataFrame based on Main Genre, timestamp, and language.
    """
    # Renaming columns
    pageviews.columns = ['Game', 'lang', 'timestamp', 'views']

    # Merging DataFrames
    merged_df = pd.merge(pageviews, game_genres, on='Game', how='left')
    merged_df.dropna(inplace=True)

    # Grouping and aggregating views
    grouped_views_df = merged_df.groupby(by=['Main Genre', 'timestamp', 'lang'], as_index=False).agg(pageviews=pd.NamedAgg(column='views', aggfunc='sum'))

    return grouped_views_df

def generate_dataset(pageviews, game_genres, interventions):
    # Renaming columns and merging data
    pageviews.columns = ['Game', 'lang', 'timestamp', 'views']
    merged_df = pd.merge(pageviews, game_genres, on='Game', how='left')
    merged_df.dropna(inplace=True)

    # Grouping and aggregating views
    grouped_views_df = merged_df.groupby(by=['Main Genre', 'timestamp', 'lang'], as_index=False).agg(pageviews=pd.NamedAgg(column='views', aggfunc='sum'))

    # Country codes to include in the analysis
    countries = {
        'France': ['fr', 'FR'],
        'Denmark': ['da', 'DK'],
        'Germany': ['de', 'DE'],
        'Italy': ['it', 'IT'],
        'Netherlands': ['nl', 'NL'],
        'Norway': ['no', 'NO'],
        'Serbia': ['sr', 'RS'],
        'Sweden': ['sv', 'SE'],
        'Korea': ['ko', 'KR'],
        'Catalonia': ['ca', 'ES'],
        'Finland': ['fi', 'FI'],
        'Japan': ['ja', 'JP'],
        'En':['En','en']
    }

    # Creating language to country map and merging dataframes
    lang_to_country_map = {lang: country for country, langs in countries.items() for lang in langs}
    interventions['Country'] = interventions['lang'].map(lang_to_country_map)
    merged_df_1 = pd.merge(grouped_views_df, interventions[['lang', 'Mobility', 'Normalcy']], on='lang', how='left')
    merged_df_1 = merged_df_1.dropna(subset=['Mobility', 'Normalcy'])

    # Adjusting 'Period' categorization
    merged_df_1['Period'] = np.select(
        [
            merged_df_1['timestamp'] < merged_df_1['Mobility'],
            (merged_df_1['timestamp'] >= merged_df_1['Mobility']) & (merged_df_1['timestamp'] < merged_df_1['Normalcy']),
            merged_df_1['timestamp'] >= merged_df_1['Normalcy']
        ],
        ['Pre-Lockdown', 'During-Lockdown', 'Post-Lockdown'],
        default='Unknown'
    )

    # Recalculating average pageviews
    avg_pageviews = merged_df_1.groupby(['Main Genre', 'lang', 'Period'])['pageviews'].mean().reset_index()

    # Pivot and calculate DiD
    pivot_avg_pageviews = avg_pageviews.pivot_table(index=['Main Genre', 'lang'], columns='Period', values='pageviews').reset_index()
    pivot_avg_pageviews.columns.name = None
    pivot_avg_pageviews['Difference'] = pivot_avg_pageviews['During-Lockdown'] - pivot_avg_pageviews['Pre-Lockdown']
    pivot_avg_pageviews['Ratio'] = (pivot_avg_pageviews['During-Lockdown'] / pivot_avg_pageviews['Pre-Lockdown'] - 1) * 100
    pivot_avg_pageviews = pivot_avg_pageviews.drop(columns=['Post-Lockdown'])

    # Filtering and merging data for different genres
    Strategy = pivot_avg_pageviews[pivot_avg_pageviews['Main Genre'] == 'Strategy']
    Action = pivot_avg_pageviews[pivot_avg_pageviews['Main Genre'] == 'Action']
    Adult = pivot_avg_pageviews[pivot_avg_pageviews['Main Genre'] == 'Adult']
    Miscellaneous = pivot_avg_pageviews[pivot_avg_pageviews['Main Genre'] == 'Miscellaneous']
    strategy_action_merged = Strategy.merge(Action, on='lang', suffixes=('_Strategy', '_Action'))
    merged_inter = Adult.merge(Miscellaneous, on='lang', suffixes=('_Adult', '_Miscellaneous'))
    comparison = strategy_action_merged.merge(merged_inter, on='lang')

    # Filtering out English language
    comparison_final = comparison[comparison['lang'] != 'en']

    # Replacing language codes with country names
    country_codes = {
        # ... existing country codes ...
        'fr': 'France',
        'da': 'Denmark',
        'de': 'Germany',
        'it': 'Italy',
        'nl': 'Netherlands',
        'no': 'Norway',
        'sr': 'Serbia',
        'sv': 'Sweden',
        'ko': 'Korea',
        'ca': 'Catalonia',
        'fi': 'Finland',
        'ja': 'Japan'
    }
    comparison_final['country'] = comparison_final['lang'].map(country_codes)

    return comparison_final

def prepare_data_for_causal_impact(grouped_views_df):
    # Convert timestamp to datetime and set as index
    def process_df(df):
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')
        df.set_index('timestamp', inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    # Filtering and pivoting data for 'Adult' and 'Action' genres
    filter_1 = grouped_views_df[
        (grouped_views_df['Main Genre'].isin(['Adult', 'Action'])) &
        (grouped_views_df['lang'].isin(['fr', 'it', 'ja', 'de']))
    ]
    group_casual_1 = process_df(
        filter_1.pivot_table(
            index=['timestamp'],
            columns=['Main Genre', 'lang'],
            values='pageviews'
        ).reset_index()
    )

    # Flatten the MultiIndex in columns for group_casual_1
    group_casual_1.columns = ['_'.join(col).strip() if col[1] else col[0] for col in group_casual_1.columns.values]

    # Filtering and pivoting data for 'Strategy' and 'Miscellaneous' genres
    filter_2 = grouped_views_df[
        (grouped_views_df['Main Genre'].isin(['Strategy', 'Miscellaneous'])) &
        (grouped_views_df['lang'].isin(['fr', 'it', 'ja', 'de']))
    ]
    group_casual_2 = process_df(
        filter_2.pivot_table(
            index=['timestamp'],
            columns=['Main Genre', 'lang'],
            values='pageviews'
        ).reset_index()
    )

    # Flatten the MultiIndex in columns for group_casual_2
    group_casual_2.columns = ['_'.join(col).strip() if col[1] else col[0] for col in group_casual_2.columns.values]

    return group_casual_1, group_casual_2

def analyze_causal_impact_before(group_casual_1, pre_period, post_period):
    # Apply CausalImpact
    ci = CausalImpact(group_casual_1['Action_fr'], pre_period, post_period)

    # Print the summary of the analysis
    print("Causal Impact Analysis for 'Action_fr'")
    print(ci.summary())

    # Plot the results
    ci.plot(panels=['original'], figsize=(15, 4))
    plt.show()

def analyze_causal_impact_with_ratio(group_casual_1, df_wikiviews):
    """
    Analyzes and plots the causal impact for the 'Action_fr' column in the provided DataFrame,
    using the ratio of this column to the 'fr' column of another DataFrame.
    """
    # Initialize a dictionary to store the results
    impact_results = {}

    # Define the pre-intervention and post-intervention periods for France
    pre_period_fr = ['2019-10-16', '2020-03-14']
    post_period_fr = ['2020-03-15', '2020-07-02']

    # Calculate the percentage ratio
    df_percentage = group_casual_1['Action_fr'] / df_wikiviews['fr']

    # Apply CausalImpact
    ci = CausalImpact(df_percentage, pre_period_fr, post_period_fr)
    impact_results['Action_fr'] = ci.summary_data

    # Print the summary of the analysis and plot the results
    print('Causal Impact Analysis for \'Action_fr\' After division by total views')
    print(ci.summary())
    ci.plot(panels=['original'], figsize=(15, 4))
    plt.show()

    return impact_results
