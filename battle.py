import pandas as pd
import numpy as np

import dash
from dash.dependencies import Output, Input, State
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.express as px
import dash_ag_grid as dag
from dash_iconify import DashIconify
import plotly.graph_objs as go
import io, os
import nltk
from nltk.probability import FreqDist
from nltk import word_tokenize
from wordcloud import WordCloud
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")
newStopWords = ['это','очень', 'нам','всё','лучше','татарстан','Text','республика','www']
russian_stopwords.extend(newStopWords)
#import pickle

try:
    app = dash.Dash(name='Tatarstan', title='Туризм в Татарстане', assets_folder='assets', external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])     
    tat_pivot = pd.read_excel(r'assets/data/Татарстан_статистика.xlsx')
    df_tat = pd.read_excel(r'assets/data/df_tatarstan_cluster_sw.xlsx')
    cluster_describe = pd.read_excel(r'assets/data/clusters_describe_sw.xlsx')
    themes_count = pd.read_excel(r'assets/data/top_themes_by_cluster_tone_date.xlsx')
except Exception as e:
    print(e)

cluster_describe = cluster_describe.T
cluster_describe['Describe'] = cluster_describe.agg(' | '.join, axis=1)
cluster_describe = cluster_describe.reset_index().drop(columns=[0,1,2,3,4])
cluster_describe = cluster_describe.rename(columns={'index':'Num_Cluster'})

dates = []
for date in tat_pivot['Год']:
    dates.append(date)

cluster_pivot = (df_tat.pivot_table(index=['Num_Cluster', 'Год_месяц'], columns='Тональность', values='Тест', aggfunc='count')
                                    .drop(columns=['skip','speech'], axis=1)
                                    .rename(columns={'negative':'Негативный отзыв', 'neutral':'Нейтральный отзыв', 'positive':'Позитивный отзыв'}))
cluster_pivot = cluster_pivot.join(df_tat.pivot_table(index=['Num_Cluster', 'Год_месяц'], values='Тест', aggfunc='count')
                                   .rename(columns={'Тест':'Всего отзывов'}))
cluster_pivot = cluster_pivot.join(df_tat.pivot_table(index=['Num_Cluster', 'Год_месяц'], values='Emoji_tone', aggfunc='count')
                                   .rename(columns={'Emoji_tone':'Постов с эмодзи'}))
cluster_pivot = cluster_pivot.join(df_tat.pivot_table(index=['Num_Cluster', 'Год_месяц'], columns='Emoji_tone', values='Тест', aggfunc='count').drop(columns=['Spam'], axis=1)
                                   .rename(columns={'Negative':'Негативные эмодзи','Neutral':'Нейтральные эмодзи','Positive':'Позитивные эмодзи'}))
cluster_pivot = cluster_pivot.reset_index()
cluster_pivot = cluster_pivot.merge(cluster_describe, on='Num_Cluster')

emoji_pivot = df_tat.pivot_table(index=['Num_Cluster', 'Год_месяц', 'Достопримечательность'], 
                            values=['Emoji_tone_pos','Emoji_tone_neg','Emoji_count_posivive','Emoji_count_negative'], 
                            aggfunc='sum', dropna=True).reset_index()



    
def unem(columns):
    em = columns[0]
    list_em = []
    for i in em:
        if not i in list_em:
            list_em.append(i)
    return list_em

emoji_pivot['Emoji_unique_pos'] = emoji_pivot[['Emoji_tone_pos']].apply(unem, axis=1)
emoji_pivot['Emoji_unique_neg'] = emoji_pivot[['Emoji_tone_neg']].apply(unem, axis=1)

table_tone = (df_tat[((df_tat['Тональность'] == 'positive') | 
                      (df_tat['Тональность'] == 'negative')) & 
                     (df_tat['Emoji_tone'] != 'Spam')]
              [['Тест','Тональность','Emoji','Emoji_tone','Год_месяц', 'Num_Cluster','Тема']].reset_index())

table_tone_neutral = (df_tat[((df_tat['Тональность'] == 'neutral')) & 
                     (df_tat['Emoji_tone'] != 'Spam')]
              [['Тест','Тональность','Emoji','Emoji_tone','Год_месяц', 'Num_Cluster','Тема']].reset_index())

date_filter_data=[{'value':val, 'label':val} for val in sorted(df_tat['Год_месяц'].unique(), reverse=True)]
cluster_filter_data=[{'value':val, 'label':val} for val in cluster_describe['Num_Cluster'].unique()]
tone_filter_data=[{'value':val, 'label':val} for val in table_tone['Тональность'].unique()]
theme_filter_data=[{'value':val, 'label':val} for val in sorted(themes_count['Тема'].unique())]


#-----------------------------------------------------------------------------#
#---------------------------------CALLBACKS-----------------------------------#

@app.callback(
    Output('avg-days-plot','figure'),
    Output('avg-days-inplace','children'),
    Output('delta-avg-days-icon','icon'),
    Output('delta-avg-days-icon','color'),
    Output('delta-avg-days','children'),
    Input('date_start_dropdown','value'),
    Input('date_end_dropdown','value')
    )
def avg_days_fig_update(date_start, date_end):
    date_ind = dates.index(date_start)
    date_now_ind = dates.index(date_end)    
    val_now = tat_pivot[tat_pivot['Год'] == date_end]['Средняя продолжительность пребывания граждан в коллективных средствах размещения'].iloc[0]
    val_prev = tat_pivot[tat_pivot['Год'] == date_start]['Средняя продолжительность пребывания граждан в коллективных средствах размещения'].iloc[0]
    delta = abs((val_now /val_prev - 1)*100).round(2) if (val_prev < val_now) else abs((1 - (val_now / val_prev)) * 100).round(2)
    delta_val = abs(val_now - val_prev)
    if val_prev < val_now:
        simbol = '+'
        delta_icon = 'feather:arrow-up-right'
        delta_icon_color = 'green'
    else: 
        simbol = '-'
        delta_icon = 'feather:arrow-down-left'
        delta_icon_color = 'red'
    
    if dates[date_now_ind] == dates[0]:
        position = 'middle right'
    elif dates[date_now_ind] == dates[-1]:
        position = 'bottom center'
    else:
        position = 'bottom center'
        
    if dates[date_ind] == dates[0]:
        position_old = 'middle right'
    elif dates[date_ind] == dates[-1]:
        position_old = 'bottom center'
    else:
        position_old = 'bottom center'
        
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tat_pivot['Год'],
            y=tat_pivot['Средняя продолжительность пребывания граждан в коллективных средствах размещения'],
            line_shape='spline',
            line=dict(color='#fff', width=3),
            #hoverinfo='skip',
            )
        )
    fig.update_traces(mode='lines')    
    fig.update_layout(
        margin={'l': 0,'r':0,'b':0,'t':0},
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        dragmode=False,
        autosize=True)
    fig.update_yaxes(
        # range=[tat_pivot['Средняя продолжительность пребывания граждан в коллективных средствах размещения'].min()+
        #        tat_pivot['Средняя продолжительность пребывания граждан в коллективных средствах размещения'].min()*0.1,
        #        tat_pivot['Средняя продолжительность пребывания граждан в коллективных средствах размещения'].max()+
        #        tat_pivot['Средняя продолжительность пребывания граждан в коллективных средствах размещения'].max()*0.05],
        showgrid=False, 
        visible=False,)
    fig.update_xaxes(showgrid=False, visible=False,)
    fig.add_scatter(
        x = [fig.data[0].x[date_ind]],
        y = [fig.data[0].y[date_ind]],
        mode='markers + text',
        text = f'{fig.data[0].y[date_ind]:.2f} дн',
        textfont = {'color':'#c7c7c7'},
        marker = {'color':'#c7c7c7', 'size':14},
        showlegend=False,
        textposition=position_old,
        hoverinfo='skip',
        )
    fig.add_scatter(
        x = [fig.data[0].x[date_now_ind]],
        y = [fig.data[0].y[date_now_ind]],
        mode='markers + text',
        text = f'{fig.data[0].y[date_now_ind]:.2f} дн',
        textfont = {'color':'#fff'},
        marker = {'color':'#fff', 'size':14},
        showlegend=False,
        textposition=position,
        hoverinfo='skip',
        )
    fig.update_annotations(opacity=0)
    return fig, val_now.round(2), delta_icon, delta_icon_color, f'{delta_val:.2f} ({delta:.2f}%) к {date_start}'


@app.callback(
    Output('people-plot','figure'),
    Output('avg-people-inplace','children'),
    Output('people-foreign-perc-value','children'),
    Output('people-foreign-value','children'),
    Output('people-rf-perc-value','children'),
    Output('people-rf-value','children'),
    Output('people-progress','value'),
    Output('delta-avg-people-icon','icon'),
    Output('delta-avg-people-icon','color'),
    Output('delta-avg-people','children'),
    Input('date_start_dropdown','value'),
    Input('date_end_dropdown','value')

    )
def people_fig_update(date_start, date_end):
    date_ind = dates.index(date_start)
    date_ind_now = dates.index(date_end)
    val_rf_now = tat_pivot[tat_pivot['Год'] == date_end]['Численность граждан Российской Федерации, размещенных в коллективных средствах размещения'].iloc[0]
    val_fgn_now  = tat_pivot[tat_pivot['Год']== date_end]['Численность иностранных граждан, размещенных в коллективных средствах размещения'].iloc[0]
    val_total_now  = val_rf_now + val_fgn_now
    
    val_rf_prev = tat_pivot[tat_pivot['Год'] == date_start]['Численность граждан Российской Федерации, размещенных в коллективных средствах размещения'].iloc[0]
    val_fgn_prev = tat_pivot[tat_pivot['Год'] == date_start]['Численность иностранных граждан, размещенных в коллективных средствах размещения'].iloc[0]
    val_total_prev  = val_rf_prev + val_fgn_prev
    
    perc_rf_now = val_rf_now / val_total_now * 100
    perc_frg_now = val_fgn_now / val_total_now * 100
    
    people_progress = int(perc_frg_now)
    
    delta = abs((val_total_now / val_total_prev - 1)*100).round(2) if (val_total_prev < val_total_now) else abs((1 - (val_total_now / val_total_prev)) * 100).round(2)
    delta_val = abs(val_total_now - val_total_prev)
    if val_total_prev < val_total_now:
        simbol = '+'
        delta_icon = 'feather:arrow-up-right'
        delta_icon_color = 'green'
    else: 
        simbol = '-'
        delta_icon = 'feather:arrow-down-left'
        delta_icon_color = 'red'
    
    if dates[date_ind] == dates[0]:
        position_old = 'top center'
    elif dates[date_ind] == dates[-1]:
        position_old = 'bottom center'
    elif dates[date_ind] == dates[-2]:
        position_old = 'bottom right'
    elif dates[date_ind] == dates[3]:
        position_old = 'bottom left'
    elif dates[date_ind] == dates[4]:
        position_old = 'top center'
    else:
        position_old = 'bottom center'
    
    if dates[date_ind_now] == dates[0]:
        position = 'top center'
    elif dates[date_ind_now] == dates[-1]:
        position = 'bottom center'
    elif dates[date_ind_now] == dates[-2]:
        position = 'bottom right'
    elif dates[date_ind_now] == dates[3]:
        position = 'bottom left'
    elif dates[date_ind_now] == dates[4]:
        position = 'top center'
    else:
        position = 'bottom center'
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tat_pivot['Год'],
            y=tat_pivot['Численность размещения общее'],
            line_shape='spline',
            line=dict(color='#fff', width=3),
            #hoverinfo='skip',
            )
        )
    fig.update_traces(mode='lines')    
    fig.update_layout(
        margin={'l': 0,'r':0,'b':0,'t':0},
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        dragmode=False,
        autosize=True)
    fig.update_yaxes(
        range=[tat_pivot['Численность размещения общее'].min()-
                tat_pivot['Численность размещения общее'].min()*0.1,
                tat_pivot['Численность размещения общее'].max()+
                tat_pivot['Численность размещения общее'].max()*0.1],
        showgrid=False, 
        visible=False,)
    fig.update_xaxes(showgrid=False, visible=False,)
    fig.add_scatter(
        x = [fig.data[0].x[date_ind]],
        y = [fig.data[0].y[date_ind]],
        mode='markers + text',
        text = f'{(fig.data[0].y[date_ind]/1000000):.2f} млн',
        textfont = {'color':'#c7c7c7'},
        marker = {'color':'#c7c7c7', 'size':14},
        showlegend=False,
        textposition=position_old,
        hoverinfo='skip',
        )
    fig.add_scatter(
        x = [fig.data[0].x[date_ind_now]],
        y = [fig.data[0].y[date_ind_now]],
        mode='markers + text',
        text = f'{(fig.data[0].y[date_ind_now]/1000000):.2f} млн',
        textfont = {'color':'#fff'},
        marker = {'color':'#fff', 'size':14},
        showlegend=False,
        textposition=position,
        hoverinfo='skip',
        )
    fig.update_annotations(opacity=0)
    return (fig, f'{val_total_now/1000000:.2f}', 
            f'{perc_frg_now:.2f}%', f'{val_fgn_now/1000000:.2f} млн', 
            f'{perc_rf_now:.2f}%', f'{val_rf_now/1000000:.2f} млн',
            people_progress, delta_icon, delta_icon_color, 
            f'{delta_val/1000000:.2f} ({delta:.2f}%) к {date_start}')


@app.callback(
    Output('profit-inplace','children'),
    Output('delta-profit-inplace-icon', 'icon'),
    Output('delta-profit-inplace-icon', 'color'),
    Output('delta-profit-inplace', 'children'),
    Output('profit-inplace-plot', 'figure'),
    
    Output('count-inplace','children'),
    Output('delta-count-inplace-icon', 'icon'),
    Output('delta-count-inplace-icon', 'color'),
    Output('delta-count-inplace','children'),
    Output('count-inplace-plot', 'figure'),
    
    Output('count-nights-inplace','children'),
    Output('delta-count-nights-inplace-icon', 'icon'),
    Output('delta-count-nights-inplace-icon', 'color'),
    Output('delta-count-nights-inplace','children'),
    Output('count-nights-inplace-plot', 'figure'),
    
    Input('date_start_dropdown','value'),
    Input('date_end_dropdown','value')
    )
def inplace_card_update(date_start, date_end): 
    date_ind = dates.index(date_start)
    date_ind_now = dates.index(date_end)
    val_profit_now = tat_pivot[tat_pivot['Год'] == date_end]['Доходы коллективных средств размещения от предоставляемых услуг без НДС, акцизов и аналогичных платежей'].iloc[0]
    val_profit_prev = tat_pivot[tat_pivot['Год'] == date_start]['Доходы коллективных средств размещения от предоставляемых услуг без НДС, акцизов и аналогичных платежей'].iloc[0]
    
    val_count_now = tat_pivot[tat_pivot['Год'] == date_end]['Число коллективных средств размещения'].iloc[0]
    val_count_prev = tat_pivot[tat_pivot['Год'] == date_start]['Число коллективных средств размещения'].iloc[0]
    
    val_nights_now = tat_pivot[tat_pivot['Год'] == date_end]['Число ночевок в коллективных средствах размещения'].iloc[0]
    val_nights_prev = tat_pivot[tat_pivot['Год'] == date_start]['Число ночевок в коллективных средствах размещения'].iloc[0]
    
    delta_profit = abs((val_profit_now / val_profit_prev - 1)*100).round(2) if (val_profit_prev < val_profit_now) else abs((1 - (val_profit_now / val_profit_prev)) * 100).round(2)
    delta_profit_val = abs(val_profit_now - val_profit_prev).round(2)
    
    if val_profit_prev < val_profit_now:
        delta_icon_profit = 'feather:arrow-up-right'
        delta_icon_color_profit = 'green'
    else: 
        delta_icon_profit = 'feather:arrow-down-left'
        delta_icon_color_profit = 'red'
        
    delta_count = abs((val_count_now / val_count_prev - 1)*100).round(2) if (val_count_prev < val_count_now) else abs((1 - (val_count_now / val_count_prev)) * 100).round(2)
    delta_count_val = abs(val_count_now - val_count_prev)
    
    if val_count_prev < val_count_now:
        delta_icon_count = 'feather:arrow-up-right'
        delta_icon_color_count = 'green'
    else: 
        delta_icon_count = 'feather:arrow-down-left'
        delta_icon_color_count = 'red'
        
    delta_nights = abs((val_nights_now / val_nights_prev - 1)*100).round(2) if (val_nights_prev < val_nights_now) else abs((1 - (val_nights_now / val_nights_prev)) * 100).round(2)
    delta_nights_val = abs(val_nights_now - val_nights_prev)
    
    if val_nights_prev < val_nights_now:
        delta_icon_nights = 'feather:arrow-up-right'
        delta_icon_color_nights = 'green'
    else: 
        delta_icon_nights = 'feather:arrow-down-left'
        delta_icon_color_nights = 'red'
    
    position_old = 'top center'
    position = position_old
    if dates[date_ind] == dates[0]:
        position_old = 'middle right'
    elif dates[date_ind] == dates[-1]:
        position_old = 'middle left'
        
    if dates[date_ind_now] == dates[0]:
        position = 'middle right'
    elif dates[date_ind_now] == dates[-1]:
        position = 'middle left'
        
    figs=[]
    
    for i in ['Доходы коллективных средств размещения от предоставляемых услуг без НДС, акцизов и аналогичных платежей',
              'Число коллективных средств размещения',
              'Число ночевок в коллективных средствах размещения']:        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=tat_pivot['Год'],
                y=tat_pivot[i],
                line_shape='spline',
                line=dict(color='#fff', width=3),
                #hoverinfo='skip',
                )
            )
        fig.update_traces(mode='lines')    
        fig.update_layout(
            margin={'l': 0,'r':0,'b':0,'t':0},
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            dragmode=False,
            autosize=True)
        fig.update_yaxes(
            range=[tat_pivot[i].min()-
                    tat_pivot[i].min()*0.1,
                    tat_pivot[i].max()+
                    tat_pivot[i].max()*0.1],
            showgrid=False, 
            visible=False,)
        fig.update_xaxes(showgrid=False, visible=False,)
        text_temp_past = f'{(fig.data[0].y[date_ind]):.0f} ед' if i == 'Число коллективных средств размещения' else f'{(fig.data[0].y[date_ind]/1000000):.2f} млн'
        text_temp_now = f'{(fig.data[0].y[date_ind_now]):.0f} ед' if i == 'Число коллективных средств размещения' else f'{(fig.data[0].y[date_ind_now]/1000000):.2f} млн'
        fig.add_scatter(
            x = [fig.data[0].x[date_ind]],
            y = [fig.data[0].y[date_ind]],
            mode='markers + text',
            text = text_temp_past,
            textfont = {'color':'#c7c7c7'},
            marker = {'color':'#c7c7c7', 'size':14},
            showlegend=False,
            textposition=position_old,
            hoverinfo='skip',
            )
        fig.add_scatter(
            x = [fig.data[0].x[date_ind_now]],
            y = [fig.data[0].y[date_ind_now]],
            mode='markers + text',
            text = text_temp_now,
            textfont = {'color':'#fff'},
            marker = {'color':'#fff', 'size':14},
            showlegend=False,
            textposition=position,
            hoverinfo='skip',
            )
        fig.update_annotations(opacity=0)
        figs.append(fig)
    
    return (f'{val_profit_now/1000000:.2f}', delta_icon_profit, delta_icon_color_profit, 
            f'{delta_profit_val/1000000:.2f} ({delta_profit:.2f}%) к {date_start}', figs[0],
            
            f'{val_count_now:.0f}', delta_icon_count, delta_icon_color_count, 
            f'{delta_count_val:.0f} ({delta_count:.0f}%) к {date_start}', figs[1],
            
            f'{val_nights_now/1000000:.2f}', delta_icon_nights, delta_icon_color_nights, 
            f'{delta_nights_val/1000000:.2f} ({delta_nights:.2f}%) к {date_start}', figs[2])


@app.callback(
    Output('tourfirm-count-all','children'),
    Output('delta-tourfirm-count-icon', 'icon'),
    Output('delta-tourfirm-count-icon', 'color'),
    Output('delta-tourfirm-count', 'children'),
    
    Output('tourfirm-tour-progress', 'value'),
    Output('tourfirm-tour-progress-abs-value', 'children'),
    Output('tourfirm-oper-progress', 'value'),
    Output('tourfirm-oper-progress-abs-value', 'children'),
    Output('tourfirm-touroper-progress', 'value'),
    Output('tourfirm-touroper-progress-abs-value', 'children'),
    
    Output('tours-count-all', 'children'),
    Output('delta-tours-count-all-icon', 'icon'),
    Output('delta-tours-count-all-icon', 'color'),
    Output('delta-tours-count-all', 'children'),
    
    Output('tours-rfrf-progress', 'value'),
    Output('tours-rfrf-progress-abs-value', 'children'),
    Output('tours-rfother-progress', 'value'),
    Output('tours-rfother-progress-abs-value', 'children'),
    Output('tours-otherrf-progress', 'value'),
    Output('tours-otherrf-progress-abs-value', 'children'),
    
    Input('date_start_dropdown','value'),
    Input('date_end_dropdown','value')
    )
def tourfirms_card_update(date_start, date_end):
    
    tour_now = tat_pivot[tat_pivot['Год'] == date_end]['Турагентская'].iloc[0]
    oper_now = tat_pivot[tat_pivot['Год'] == date_end]['Туроператорская'].iloc[0]
    touroper_now = tat_pivot[tat_pivot['Год'] == date_end]['Туроператорская и турагентская'].iloc[0]
    firm_all_now = tour_now + oper_now + touroper_now   
    
    tour_past = tat_pivot[tat_pivot['Год'] == date_start]['Турагентская'].iloc[0]
    oper_past = tat_pivot[tat_pivot['Год'] == date_start]['Туроператорская'].iloc[0]
    touroper_past = tat_pivot[tat_pivot['Год'] == date_start]['Туроператорская и турагентская'].iloc[0]
    firm_all_past = tour_past + oper_past + touroper_past 
    
    tour_perc_now = int(tour_now / firm_all_now * 100)
    oper_perc_now = int(oper_now / firm_all_now * 100)
    touroper_perc_now = int(touroper_now / firm_all_now * 100)

    delta_firm = abs((firm_all_now / firm_all_past - 1)*100).round(2) if (firm_all_past < firm_all_now) else abs((1 - (firm_all_now / firm_all_past)) * 100).round(2)
    delta_firm_val = abs(firm_all_now - firm_all_past)
    if firm_all_past < firm_all_now:
        delta_icon_firm = 'feather:arrow-up-right'
        delta_icon_color_firm = 'green'
    else: 
        delta_icon_firm = 'feather:arrow-down-left'
        delta_icon_color_firm = 'red'
    
    rfrf_now = tat_pivot[tat_pivot['Год'] == date_end]['Турпакетов Гражданам России по территории России'].iloc[0]
    rfother_now = tat_pivot[tat_pivot['Год'] == date_end]['Турпакетов Гражданам России по другим странам'].iloc[0]
    otherrf_now = tat_pivot[tat_pivot['Год'] == date_end]['Турпакетов Гражданам других стран по территории России'].iloc[0]
    tours_all_now = rfrf_now + rfother_now + otherrf_now    
    
    rfrf_past = tat_pivot[tat_pivot['Год'] == date_start]['Турпакетов Гражданам России по территории России'].iloc[0]
    rfother_past = tat_pivot[tat_pivot['Год'] == date_start]['Турпакетов Гражданам России по другим странам'].iloc[0]
    otherrf_past = tat_pivot[tat_pivot['Год'] == date_start]['Турпакетов Гражданам других стран по территории России'].iloc[0]
    tours_all_past = rfrf_past + rfother_past + otherrf_past
    
    rfrf_perc_now = int(rfrf_now / tours_all_now * 100)
    rfother_perc_now = int(rfother_now / tours_all_now * 100)
    otherrf_perc_now = int(otherrf_now / tours_all_now * 100)
    
    delta_tours = abs((tours_all_now / tours_all_past - 1)*100).round(2) if (tours_all_past < tours_all_now) else abs((1 - (tours_all_now / tours_all_past)) * 100).round(2)
    delta_tours_val = abs((tours_all_now - tours_all_past)/1000)
    if tours_all_past < tours_all_now:
        delta_icon_tours = 'feather:arrow-up-right'
        delta_icon_color_tours = 'green'
    else: 
        delta_icon_tours = 'feather:arrow-down-left'
        delta_icon_color_tours = 'red'
        
    return (firm_all_now, delta_icon_firm, delta_icon_color_firm, 
            f'{delta_firm_val:.0f} ({delta_firm:.0f}%) к {date_start}',
            
            tour_perc_now, f'{tour_now} единиц', 
            oper_perc_now, f'{oper_now} единиц',
            touroper_perc_now, f'{touroper_now} единиц',
            
            f'{tours_all_now/1000:.1f}', delta_icon_tours, delta_icon_color_tours, 
            f'{delta_tours_val:.1f} ({delta_tours:.0f}%) к {date_start}',
            
            rfrf_perc_now, f'{rfrf_now/1000:.1f} тыс шт.', 
            rfother_perc_now, f'{rfother_now/1000:.1f} тыс шт.', 
            otherrf_perc_now, f'{otherrf_past} штук')
       

@app.callback(
    Output('most-famous-pie-in-the-world','figure'),
    Output('pie-positive-value','children'),
    Output('pie-positive-perc','children'),
    
    Output('pie-neutral-value','children'),
    Output('pie-neutral-perc','children'),
    
    Output('pie-negative-value','children'),
    Output('pie-negative-perc','children'),
    
    Input('filter-theme','value'),
    Input('filter-cluster','value'),
    Input('filter-date','value'),
    )
def pie_upadte(theme, cluster, date):    
    fig = go.Figure()
    df_tat_temp = df_tat
    if theme:
        df_tat_temp = df_tat_temp[df_tat_temp['Тема'].isin(theme)]
    if cluster:
        df_tat_temp = df_tat_temp[df_tat_temp['Num_Cluster'].isin(cluster)]
    if date:
        df_tat_temp = df_tat_temp[df_tat_temp['Год_месяц'].isin(date)]

    df_pivot = df_tat_temp[(df_tat_temp['Тональность']=='negative') |
                           (df_tat_temp['Тональность']=='neutral') |
                           (df_tat_temp['Тональность']=='positive')].pivot_table(values='Тест', index='Тональность', aggfunc='count').reset_index()
            
    total = df_pivot['Тест'].sum()
    pos_val = df_pivot[df_pivot['Тональность']=='positive']['Тест'].sum()
    pos_perc = (pos_val / total) * 100
    neu_val =df_pivot[df_pivot['Тональность']=='neutral']['Тест'].sum()
    neu_perc = (neu_val / total) * 100
    neg_val = df_pivot[df_pivot['Тональность']=='negative']['Тест'].sum()
    neg_perc = (neg_val / total) * 100
        
    fig = px.pie(df_pivot, 
                 values='Тест',
                 names='Тональность', 
                 hole=.7, 
                 color='Тональность',
                 color_discrete_map={'positive':'rgb(144, 238, 144)',
                                     'neutral':'rgb(220, 220, 220)',
                                     'negative':'rgb(255, 99, 71)'}
                 )
    fig.update_traces(hovertemplate=None, textposition='outside') 
    fig.update_yaxes(showgrid=False, visible=False)
    fig.update_xaxes(showgrid=False, visible=True)
    fig.update_layout(
            xaxis_title='',
            margin={'l': 0,'r':0,'b':0,'t':0},
            showlegend=False,
            font_color="white",
            hoverlabel = dict(font=dict(color='black')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            dragmode=False,
            autosize=True,
            uniformtext_minsize=8, 
            uniformtext_mode='hide',
            annotations=[dict(text=f'{total} шт.', x=0.5, y=0.5, font_size=15, showarrow=False)]
    )
    
    return (fig, 
            pos_val, f'{pos_perc:.1f} %', 
            neu_val, f'{neu_perc:.1f} %', 
            neg_val, f'{neg_perc:.1f} %')


@app.callback(
    [[Output('cluster-pivot-table','rowData')],
    Output('cluster-pivot-table','columnDefs')],
    Input('famous_type_radio','value'),
    Input('filter-theme','value'),
    Input('filter-cluster','value'),
    Input('filter-date','value'),
    )
def clusters_pivot_table_update(famous, themes,cluster, date_filter):
    if famous == 'famous':
        themes_temp = themes_count
        if cluster:
            themes_temp = themes_temp[themes_temp['Num_Cluster'].isin(cluster)]
        if date_filter:
            themes_temp = themes_temp[themes_temp['Год_месяц'].isin(date_filter)]
        cluster_pivot1 = themes_temp.pivot_table(index=['Тема'], values='Кол-во', aggfunc='sum').reset_index()
        cluster_pivot2 = themes_temp.pivot_table(index=['Тема'], columns='Тональность', values='Кол-во', aggfunc='sum').reset_index()
        cluster_pivot_date = pd.merge(cluster_pivot1,cluster_pivot2, on='Тема').sort_values('Кол-во', ascending=False)
        cluster_pivot_date['id'] = cluster_pivot_date.index
        cluster_pivot_date = cluster_pivot_date[cluster_pivot_date['Кол-во'] > 0]
        columnDefs  = [
            {
            'headerName': '№',
            'children': [
                {'field':'id', 'headerName':'', 'width': 46},
                ],
            },
            {
            'headerName': '',
            'children': [
                {'field':'Тема', 'headerName':'Тематика', 'width': 166, 'minWidth':142},
                ],
            },
            {
            'headerName': 'Кол-во отзывов',
            'children': [                                        
                {'field':'Кол-во', 'headerName':'Всего', 'width': 74},
                ]
            },
            {
            'headerName': 'Тональность отзыва',
            'children': [                                                                            
                {'field':'positive', 'headerName':'Позитивный', 'width': 111}, 
                {'field':'negative', 'headerName':'Негативный', 'width': 110},
                {'field':'neutral', 'headerName':'Нейтральный', 'width': 117},                
                {'field':'speech', 'headerName':'Речь', 'width': 67},
                {'field':'skip', 'headerName':'Прочее', 'width': 83},
                ],
            },            
            ]
    else:
        df_tat_temp=df_tat
        if themes:
            df_tat_temp = df_tat_temp[df_tat_temp['Тема'].isin(themes)]
        if date_filter:
            df_tat_temp = df_tat_temp[df_tat_temp['Год_месяц'].isin(date_filter)]
        cluster_pivot_date = (df_tat_temp.pivot_table(index=['Num_Cluster'], columns='Тональность', values='Тест', aggfunc='count')
                                            #.drop(columns=['skip','speech'], axis=1)
                                            .rename(columns={'negative':'Негативный отзыв', 'neutral':'Нейтральный отзыв', 'positive':'Позитивный отзыв'}))
        cluster_pivot_date = cluster_pivot_date.join(df_tat_temp.pivot_table(index=['Num_Cluster'], values='Тест', aggfunc='count')
                                           .rename(columns={'Тест':'Всего отзывов'}))
        cluster_pivot_date = cluster_pivot_date.join(df_tat_temp.pivot_table(index=['Num_Cluster'], values='Emoji_tone', aggfunc='count')
                                           .rename(columns={'Emoji_tone':'Постов с эмодзи'}))
        cluster_pivot_date = cluster_pivot_date.join(df_tat_temp.pivot_table(index=['Num_Cluster'], columns='Emoji_tone', values='Тест', aggfunc='count')#.drop(columns=['Spam'], axis=1)
                                           .rename(columns={'Negative':'Негативные эмодзи','Neutral':'Нейтральные эмодзи','Positive':'Позитивные эмодзи'}))
        cluster_pivot_date['Describe'] = cluster_describe['Describe']
        cluster_pivot_date['id'] = cluster_pivot_date.index
        columnDefs  = [
            {
            'headerName': '№',
            'children': [
                {'field':'id', 'headerName':'', 'width': 38},
                ],
            },
            {
            'headerName': 'Кластер',
            'children': [
                {'field':'Describe', 'headerName':'', 'width': 278, 'minWidth':142},
                ],
            },
            {
            'headerName': 'Кол-во отзывов',
            'children': [                                        
                {'field':'Всего отзывов', 'headerName':'', 'width': 72, 'maxWidth':72 },
                ]
            },
            {
            'headerName': 'Тональность отзыва',
            'children': [                                                                            
                {'field':'Негативный отзыв', 'headerName':'-', 'width': 47}, 
                {'field':'Нейтральный отзыв', 'headerName':'0', 'width': 62},
                {'field':'Позитивный отзыв', 'headerName':'+', 'width': 55},
                ],
            },
            {
            'headerName': 'Кол-во эмодзи',
            'children': [
                {'field':'Постов с эмодзи', 'headerName':'', 'width': 67, 'maxWidth':70},                   
                ],
            },
            {
            'headerName': 'Тональность эмодзи',
            'children': [                                                                            
                {'field':'Негативные эмодзи', 'headerName':'-', 'width': 46}, 
                {'field':'Нейтральные эмодзи', 'headerName':'0', 'width': 55},
                {'field':'Позитивные эмодзи', 'headerName':'+', 'width': 56},
                ],
            }
            ]
    return [[cluster_pivot_date.to_dict('records')], columnDefs]


@app.callback(
    Output('voice-by-clusters-plot', 'figure'),
    Output('header-voices','children'),
    Input('date_end_dropdown','value'),
    Input('cluster_type_radio','value'),
    Input('famous_type_radio','value'),
    Input('filter-cluster','value'),
    Input('filter-tone','value'),
    Input('filter-date','value'),
    Input('filter-theme','value'),
    )
def clust_polt_update(date_end, type_graph, famous_type, cluster, tone, date_filter, theme):    
    header='По кластерам'
    fig = go.Figure()

    if type_graph == 'all' and famous_type == 'cluster':  
        cluster_pivot_df_temp = cluster_pivot
        if cluster:
            cluster_pivot_df_temp = cluster_pivot_df_temp[cluster_pivot_df_temp['Num_Cluster'].isin(cluster)]
        if tone:
            if ('positive' in tone) and ('negative' in tone):
                cluster_pivot_temp = cluster_pivot_df_temp.pivot_table(index='Год_месяц', values=['Всего отзывов'], aggfunc='sum').reset_index()    
            if 'positive' in tone:
                cluster_pivot_temp = cluster_pivot_df_temp.pivot_table(index='Год_месяц', values=['Позитивный отзыв'], aggfunc='sum').reset_index().rename(columns={'Позитивный отзыв':'Всего отзывов'})
            else:
                cluster_pivot_temp = cluster_pivot_df_temp.pivot_table(index='Год_месяц', values=['Негативный отзыв'], aggfunc='sum').reset_index().rename(columns={'Негативный отзыв':'Всего отзывов'})            
        else:
            cluster_pivot_temp = cluster_pivot_df_temp.pivot_table(index='Год_месяц', values=['Всего отзывов'], aggfunc='sum').reset_index()
            
        #cluster_pivot_temp = cluster_pivot.pivot_table(values=['Всего отзывов'], columns=['Num_Cluster','Год_месяц'], aggfunc='sum').reset_index().T.reset_index()[1:]
        fig = px.bar(cluster_pivot_temp, 
                     x=cluster_pivot_temp['Год_месяц'], 
                     y=cluster_pivot_temp['Всего отзывов'],
                     text=cluster_pivot_temp['Всего отзывов'],
                     #color=cluster_pivot_temp['Num_Cluster']
                     )
        
        fig.update_traces(hovertemplate=None, 
                          marker_color = '#c1d7f7',
                          texttemplate='%{text:.2s}', 
                          textposition='outside') 
        fig.update_yaxes(showgrid=False, visible=False,
                         range=[0,
                               cluster_pivot_temp['Всего отзывов'].max()+
                               cluster_pivot_temp['Всего отзывов'].max()*0.2])
        fig.update_xaxes(showgrid=False, visible=True)
        fig.update_layout(
            xaxis_title='',
            margin={'l': 0,'r':0,'b':0,'t':0},
            showlegend=False,
            font_color="white",
            hoverlabel = dict(font=dict(color='black')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            dragmode=False,
            autosize=True,
            uniformtext_minsize=8, 
            uniformtext_mode='hide'
            )
    elif type_graph == 'cluster' and famous_type == 'cluster':
        cluster_pivot_temp = cluster_pivot.pivot_table(index='Год_месяц', values=['Всего отзывов'], columns='Num_Cluster', aggfunc='sum').reset_index()
        cluster_pivot_temp.columns = cluster_pivot_temp.columns.droplevel()
        cluster_pivot_temp = cluster_pivot_temp.rename(columns={'':'Год_месяц'})
        
        color_list = ['#fff','#97B2DE','#656668','#35A792','#B4D4BF','#A9A9A9','#D5D5D5','#DB3E4D','#E36E7C','#FADEA1','#BEA096','#99FF99','#FF99CC','#6666FF', '#60FDFD']
        for color in color_list:
            col = color_list.index(color)
            fig.add_trace(
                go.Scatter(
                    x=cluster_pivot_temp['Год_месяц'],
                    y=cluster_pivot_temp[col],
                    line_shape='spline',
                    line=dict(color=color, width=3),
                    #hoverinfo='skip',
                    )
                )
        
        fig.update_traces(mode='lines', hovertemplate=None)    
        fig.update_layout(
            margin={'l': 0,'r':0,'b':0,'t':0},
            showlegend=True,
            font_color="white",
            hoverlabel = dict(font=dict(color='black')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            dragmode=False,
            autosize=True,
            legend=dict(
                    orientation="h",
                    entrywidth=15,
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    
                ))
        for i in cluster_describe['Num_Cluster']:
            fig.data[int(i)].name = str(i)    
        fig.update_yaxes(showgrid=False, visible=False)
        fig.update_xaxes(showgrid=False, visible=True)
    
        fig.update_annotations(opacity=0)
    else:   
        themes_count_temp = themes_count
        if cluster:
            themes_count_temp = themes_count_temp[themes_count_temp['Num_Cluster'].isin(cluster)]
        if tone:
            themes_count_temp = themes_count_temp[themes_count_temp['Тональность'].isin(tone)]
        top10 = themes_count_temp.pivot_table(values='Кол-во', index = ['Тема'], aggfunc='sum').reset_index().sort_values('Кол-во', ascending=False)[:10]['Тема']
        cluster_pivot_temp =  themes_count_temp[themes_count_temp['Тема'].isin(top10)].pivot_table(values='Кол-во', index = ['Тема','Год_месяц'], aggfunc='sum').reset_index().sort_values('Кол-во', ascending=False)

        #color_list = ['#fff','#97B2DE','#656668','#35A792','#B4D4BF','#A9A9A9',
        #'#D5D5D5','#DB3E4D','#E36E7C','#FADEA1','#BEA096','#99FF99','#FF99CC']
        for th in top10:    
            fig.add_trace(
                    go.Scatter(
                        x=cluster_pivot_temp[(cluster_pivot_temp['Тема']==th)].sort_values('Год_месяц')['Год_месяц'],
                        y=cluster_pivot_temp[(cluster_pivot_temp['Тема']==th)].sort_values('Год_месяц')['Кол-во'],                
                        line_shape='spline',
                        mode='lines',    
                        #line=dict(color=color, width=3),
                        #hoverinfo='skip',                
                        )
                    )
        for i in range(len(cluster_pivot_temp['Тема'].unique())):
            fig.data[i].name = cluster_pivot_temp['Тема'].unique()[i]   
        fig.update_yaxes(showgrid=False, visible=False)  
        fig.update_xaxes(showgrid=False, visible=True)       
        fig.update_traces(mode='lines', hovertemplate=None)    
        fig.update_layout(
            margin={'l': 0,'r':0,'b':0,'t':0},
            showlegend=True,
            font_color="white",
            hoverlabel = dict(font=dict(color='black')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            dragmode=False,
            autosize=True,
            legend=dict(
                orientation="h",
                entrywidth=150,
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font = dict(size = 10, color='white')
                ))
        fig.update_annotations(opacity=0)
        header='По тематике'
    return fig, header

@app.callback(
    Output('voice-by-tone-plot', 'figure'),    
    Input('filter-cluster','value'),
    Input('filter-tone','value'),
    Input('famous_type_radio','value'),
    Input('filter-date','value'),
    Input('filter-theme','value'),
    )
def tone_plot_update(cluster, tone, famous,date_filter,theme):
    if famous == 'cluster':
        if cluster:
            cluster_pivot_temp = (cluster_pivot[cluster_pivot['Num_Cluster'].isin(cluster)]
                                  .pivot_table(index='Год_месяц', values=['Негативный отзыв','Позитивный отзыв'], aggfunc='sum')
                                                        .reset_index().rename(columns={'Негативный отзыв':'Негативные','Позитивный отзыв':'Позитивные'}))
        else:
            cluster_pivot_temp = (cluster_pivot.pivot_table(index='Год_месяц', values=['Негативный отзыв','Позитивный отзыв'], aggfunc='sum')
                                                        .reset_index().rename(columns={'Негативный отзыв':'Негативные','Позитивный отзыв':'Позитивные'}))
    
        fig = go.Figure()
        
        def neg():
            fig.add_trace(
                go.Scatter(
                    x=cluster_pivot_temp['Год_месяц'],
                    y=cluster_pivot_temp['Негативные'],
                    fill='tozeroy',
                    line_shape='spline',
                    line=dict(color='#ff6666', width=3),
                    #hoverinfo='skip',
                    )
                )
        def pos():
            fig.add_trace(
                go.Scatter(
                    x=cluster_pivot_temp['Год_месяц'],
                    y=cluster_pivot_temp['Позитивные'],
                    fill='tozeroy',
                    line_shape='spline',
                    line=dict(color='#fff', width=3),
                    #hoverinfo='skip',
                    )
                )
        if len(tone)==1:
            if tone[0] == 'negative': 
                neg()
                fig.data[0].name = 'Негативные'
            if tone[0] == 'positive': 
                pos()  
                fig.data[0].name = 'Позитивные'
        elif (not tone) or (len(tone)==2):
            neg()
            pos()
            fig.data[0].name = 'Негативные'
            fig.data[1].name = 'Позитивные'            

        fig.update_yaxes(
            showgrid=False, 
            visible=False,)
        fig.update_xaxes(showgrid=False, visible=True, title_font_color='white',)
        fig.update_yaxes(showgrid=False, visible=False)  
        fig.update_traces(mode='lines', hovertemplate=None)
        fig.update_layout(
            title='Отзывы по тональности',
            title_font_color='white',
            font_color="white",
            hoverlabel = dict(font=dict(color='black')),
            margin={'l': 0,'r':0,'b':0,'t':0},
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            dragmode=False,
            autosize=True,
            legend=dict(
                    orientation="h",
                    entrywidth=70,
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=0.9
                ))
        fig.update_annotations(opacity=0)
    else:         
        df_tat_temp = df_tat[df_tat['Тональность'].isin(['positive','negative'])]
        if theme:
            df_tat_temp = df_tat_temp[df_tat_temp['Тема'].isin(theme)]
        if cluster:
            df_tat_temp = df_tat_temp[df_tat_temp['Num_Cluster'].isin(cluster)]  
        
        cluster_pivot_temp2 = df_tat_temp.pivot_table(index=['Достопримечательность','Тональность','Год_месяц'], values=['Тест'], aggfunc='count').reset_index()
        
        if not('Тест' in cluster_pivot_temp2.columns):
            cluster_pivot_temp2['Тест'] = 0
        cluster_pivot_temp2 = cluster_pivot_temp2.pivot_table(index='Год_месяц', values='Тест', columns='Тональность', aggfunc='sum').reset_index()
        
        if len(cluster_pivot_temp2) == 0:
            cluster_pivot_temp2['Год_месяц'] = cluster_pivot['Год_месяц'].unique()
            cluster_pivot_temp2['negative'] = 0        
            cluster_pivot_temp2['positive'] = 0
            
        if not ('negative' in cluster_pivot_temp2.columns):
            cluster_pivot_temp2['negative']=0
        if not ('positive' in cluster_pivot_temp2.columns):
            cluster_pivot_temp2['positive']=0   
        cluster_pivot_temp2 = cluster_pivot_temp2.sort_values('Год_месяц')

        fig = go.Figure()
        
        def neg1():
            fig.add_trace(
                go.Scatter(
                    x=cluster_pivot_temp2['Год_месяц'],
                    y=cluster_pivot_temp2['negative'],                
                    fill='tozeroy',
                    line_shape='spline',
                    line=dict(color='#ff6666', width=3),
                    mode='lines',
                    #line=dict(color=color, width=3),
                    #hoverinfo='skip',                
                    )
                )
        def pos1():
            fig.add_trace(
                go.Scatter(
                    x=cluster_pivot_temp2['Год_месяц'],
                    y=cluster_pivot_temp2['positive'],       
                    fill='tozeroy',
                    line_shape='spline',
                    line=dict(color='#fff', width=3),
                    mode='lines',
                    #line=dict(color=color, width=3),
                    #hoverinfo='skip',                
                    )
                )
        
        if len(tone)==1:
            if tone[0] == 'negative': 
                neg1()
                fig.data[0].name = 'Негативные'
            if tone[0] == 'positive': 
                pos1()  
                fig.data[0].name = 'Позитивные'
        elif (not tone) or (len(tone)==2):
            neg1()
            pos1()
            fig.data[0].name = 'Негативные'
            fig.data[1].name = 'Позитивные'
            
        fig.update_traces(mode='lines', hovertemplate=None) 
        fig.update_yaxes(showgrid=False, visible=False)  
        fig.update_xaxes(showgrid=False, visible=True)   
        fig.update_layout(
            title='Темы по тональности',
            title_font_color='white',
            font_color="white",
            hoverlabel = dict(font=dict(color='black')),
            margin={'l': 0,'r':0,'b':0,'t':0},
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            dragmode=False,
            autosize=True,
            legend=dict(
                    orientation="h",
                    entrywidth=70,
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=0.9
                ))


            
    return fig

@app.callback(
    Output('emoji-by-tone-plot', 'figure'),    
    Input('filter-cluster','value'),
    Input('filter-tone','value'),
    Input('filter-theme','value'),
    Input('famous_type_radio','value'),
    )
def emoji_plot_update(cluster, tone, theme, famous):
    if famous == 'famous':
        themes_count_temp = themes_count
        #top10_themes = themes_count_temp.pivot_table(values='Кол-во', index = 'Тема', aggfunc='sum').reset_index().sort_values('Кол-во', ascending=False)[:10]['Тема']        
        df_tat_temp = df_tat[df_tat['Тональность'].isin(['positive','negative'])]
        if theme:
            df_tat_temp = df_tat_temp[df_tat_temp['Тема'].isin(theme)]
        if cluster:
            df_tat_temp = df_tat_temp[df_tat_temp['Num_Cluster'].isin(cluster)]
        try:
            cluster_pivot1 = df_tat_temp.pivot_table(index=['Достопримечательность','Тональность','Год_месяц'], values=['Тест'], columns=['Emoji_tone'], aggfunc='count').reset_index()        
            cluster_pivot_temp = cluster_pivot1.sort_values('Год_месяц')
            cluster_pivot_temp.columns = cluster_pivot_temp.columns.droplevel() 
            columns_new = ['Тема', 'Тональность', 'Год_месяц', 'negative', 'neutral','positive','spam']
            cluster_pivot_temp.columns = columns_new
            cluster_pivot_temp = cluster_pivot_temp.pivot_table(index='Год_месяц', values=['negative','positive'], aggfunc='sum').reset_index()
        except:
            cluster_pivot_temp = pd.DataFrame()
            cluster_pivot_temp['Год_месяц'] = cluster_pivot['Год_месяц'].unique()
            cluster_pivot_temp['negative'] = 0        
            cluster_pivot_temp['positive'] = 0
        xx = 'negative'
        xy = 'positive'
        xxx = 'Эмодзи в темах по тональности без спама'
    else:
        xx = 'Негативные'
        xy = 'Позитивные'
        xxx = 'Эмодзи по тональности без спама'
        if cluster:
            cluster_pivot_temp = (cluster_pivot[cluster_pivot['Num_Cluster'].isin(cluster)]
                                  .pivot_table(index='Год_месяц', values=['Негативные эмодзи','Позитивные эмодзи'], aggfunc='sum')
                                                        .reset_index().rename(columns={'Негативные эмодзи':'Негативные','Позитивные эмодзи':'Позитивные'}))
        else:
            cluster_pivot_temp = (cluster_pivot.pivot_table(index='Год_месяц', values=['Негативные эмодзи','Позитивные эмодзи'], aggfunc='sum')
                                                        .reset_index().rename(columns={'Негативные эмодзи':'Негативные','Позитивные эмодзи':'Позитивные'}))
    fig = go.Figure()    
   
    def neg():
        fig.add_trace(
            go.Scatter(
                x=cluster_pivot_temp['Год_месяц'],
                y=cluster_pivot_temp[xx],
                fill='tozeroy',
                line_shape='spline',
                line=dict(color='#ff6666', width=3),
                #hoverinfo='skip',
                )
            )
    def pos():
        fig.add_trace(
            go.Scatter(
                x=cluster_pivot_temp['Год_месяц'],
                y=cluster_pivot_temp[xy],
                fill='tonexty',
                line_shape='spline',
                line=dict(color='#fff', width=3),
                #hoverinfo='skip',
                )
            )
    

    if len(tone)==1:
        if tone[0] == 'negative': 
            neg()
            fig.data[0].name = 'Негативные'
        if tone[0] == 'positive': 
            pos()  
            fig.data[0].name = 'Позитивные'
    elif (not tone) or (len(tone)==2):
        neg()
        pos()
        fig.data[0].name = 'Негативные'
        fig.data[1].name = 'Позитивные'
    fig.update_traces(mode='lines', hovertemplate=None)
    fig.update_layout(
        title=xxx,
        title_font_color="white",
        font_color="white",
        hoverlabel = dict(font=dict(color='black')),
        margin={'l': 0,'r':0,'b':0,'t':0},
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        dragmode=False,
        autosize=True,
        legend=dict(
            orientation="h",
            entrywidth=70,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.9
            ))
      
    fig.update_yaxes(
        showgrid=False, 
        visible=False,)
    fig.update_xaxes(showgrid=False, visible=True,)

    fig.update_annotations(opacity=0)
    return fig


@app.callback(
    Output('positive-emoji-stuck','children'),
    Output('negative-emoji-stuck','children'),
    Output('header-positive-emoji-container','children'),
    Output('header-negative-emoji-container','children'),
    Input('filter-cluster','value'),
    Input('filter-theme','value'),
    Input('filter-date','value'),
    Input('famous_type_radio','value'),
    )
def emoji_list_update(cluster,theme,date, famous):
    emoji_pivot_temp = emoji_pivot

    if cluster:
        emoji_pivot_temp = emoji_pivot_temp[emoji_pivot_temp['Num_Cluster'].isin(cluster)]
    if theme:
        emoji_pivot_temp = emoji_pivot_temp[emoji_pivot_temp['Достопримечательность'].isin(theme)]
    if date:
        emoji_pivot_temp = emoji_pivot_temp[emoji_pivot_temp['Год_месяц'].isin(date)]
    # if theme:
    #     emoji_pivot_temp = emoji_pivot_temp[emoji_pivot_temp['Num_Cluster'].isin(cluster)]
    pos_count = emoji_pivot_temp['Emoji_count_posivive'].sum()
    neg_count = emoji_pivot_temp['Emoji_count_negative'].sum()
    pos_list = "".join(set(str(emoji_pivot_temp[['Emoji_tone_pos']]
                               .apply(unem, axis=1)
                               .sum())
                           .translate({ord(i):None for i in '[]", '})
                           .translate({ord(i):None for i in "'"})))
    neg_list = "".join(set(str(emoji_pivot_temp[['Emoji_tone_neg']]
                               .apply(unem, axis=1)
                               .sum())
                           .translate({ord(i):None for i in '[]", '})
                           .translate({ord(i):None for i in "'"})))
        
    return (f'{pos_count} ед  ::  {pos_list}', f'{neg_count} ед  ::  {neg_list}', 
           'Положительные эмодзи', 'Отрицательные эмодзи')


@app.callback(
    Output('wordcloud','figure'),
    Input('filter-date','value'),
    Input('filter-cluster','value'),
    Input('filter-tone','value'),
    Input('filter-theme','value'),
    Input('famous_type_radio','value'),
    )
def wordcloud_update(date_filter, cluster, tone, theme, famous):
    
    if famous == 'famous':
        themes_count_temp = themes_count
        if cluster:
            themes_count_temp = themes_count_temp[themes_count_temp['Num_Cluster'].isin(cluster)]
        if tone:
            themes_count_temp = themes_count_temp[themes_count_temp['Тональность'].isin(tone)]
        if date_filter:
            themes_count_temp = themes_count_temp[themes_count_temp['Год_месяц'].isin(date_filter)]
        text = themes_count_temp.pivot_table(values='Кол-во', index = 'Тема', aggfunc='sum').reset_index().sort_values('Кол-во', ascending=False)[:20]['Тема']
        word_str = []        
        for i in text:
            word_str.append(i)
        word_str = word_str[:30]
    else:
        text = df_tat
        if cluster:
            text = text[text['Num_Cluster'].isin(cluster)]
        if tone:
            text = text[text['Тональность'].isin(tone)]
        if theme:
            text = text[text['Тема'].isin(theme)]
        if date_filter:
            text = text[text['Год_месяц'].isin(date_filter)]
        text = text['text_lemm']
        txt = nltk.Text(text)
        tokens = word_tokenize(str(txt))
        word_str = []
        for i in (["".join((i)) for i in FreqDist(tokens)][:40]):
            if not len(i)<=2:
                word_str.append(i)    
    word_str = " ".join(word_str)
    if word_str=='Text ...':
        word_str = 'отсутствуют данные'
    wordcloud = WordCloud(
        background_color ="rgba(255, 255, 255, 0)", 
        mode="RGBA",
        width = 1100,
        height = 500,
        stopwords=russian_stopwords,
        #colormap='gist_gray_r'
        color_func=lambda *args, **kwargs: "white",
        font_path=r'assets/css/fonts/Geologica-Thin.ttf'
        ).generate(word_str)
    fig = px.imshow(wordcloud)
    fig.update_layout(
        title='',
        margin={'l': 0,'r':0,'b':0,'t':0},
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        dragmode=False,
        autosize=True)
    fig.update_traces(hovertemplate=None, hoverinfo='skip') 
    fig.update_yaxes(showgrid=False, visible=False)
    fig.update_xaxes(showgrid=False, visible=False)
        
    return fig


@app.callback(
    Output('voice-text-table','rowData'),
    Input('filter-cluster','value'),
    Input('filter-tone','value'),
    Input('filter-theme','value'),
    Input('filter-date','value'),
    )
def voice_table_update(cluster, tone, theme, date_filter):
    table = table_tone
    table_neutral = table_tone_neutral
    if cluster:
        table = table[table['Num_Cluster'].isin(cluster)]
        table_neutral = table_neutral[table_neutral['Num_Cluster'].isin(cluster)]
    if tone:
        table = table[table['Тональность'].isin(tone)]
        table_neutral = table_neutral[table_neutral['Тональность'].isin(tone)]
    if theme:
        table = table[table['Тема'].isin(theme)]
        table_neutral = table_neutral[table_neutral['Тема'].isin(theme)]
    if date_filter:
        table = table[table['Год_месяц'].isin(date_filter)]  
        table_neutral = table_neutral[table_neutral['Год_месяц'].isin(date_filter)]
    if len(table) == 0:
        table = table_neutral
    return table.to_dict('records') 

@app.callback(
    Output('cluster_type_radio-container','style'),
    Output('header-cluster-pivot-table','children'),
    Input('famous_type_radio','value'),
    Input('cluster_type_radio-container','style'),
    )
def cluster_selector_hide(famous, style):    
    if famous =='famous':
        style['display'] = 'none' 
        header = 'По тематике'
    else: 
        style['display'] = 'block'
        header = 'По кластерам'
    return style, header


@app.callback(
    Output('filter-theme','data'),
    
    Input('filter-cluster','value'),
    Input('filter-tone','value'),
    Input('filter-date','value'),
    )
def crossfilters(cluster, tone, date):    
    if not (any([cluster, tone, date])):
        return theme_filter_data
    fdf = themes_count
    if date:
        fdf = fdf[fdf['Год_месяц'].isin(date)]
    if tone:
        fdf = fdf[fdf['Тональность'].isin(tone)]
    if cluster:
        fdf = fdf[fdf['Num_Cluster'].isin(cluster)] 
    fdf = fdf[fdf['Кол-во']>0]
    theme_data=[{'value':val, 'label':val} for val in sorted(fdf['Тема'].unique())]    
    return theme_data
#-----------------------------END OF CALLBACKS--------------------------------#
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
#--------------------------------THE LAYOUT-----------------------------------#

app.layout = html.Div(className='main-layout',
children = [
    dcc.Store(id='main_store'),
    dmc.Header(
        className='header-layout',
        height='5vh', 
               children=[
                      html.Div("Туризм в Татарстане", 
                               style={'padding-left':'30px',
                                      'color':'white', 'font-size': 'xx-large'}),                   
                      html.Div([
                          html.Div('Все данные идут с доковидным годом 2019, но вы можете выбрать периоды сравнения',
                                   style={'opacity':'70%',
                                          'color':'white', 'font-size': 'large',
                                          'padding-right':'5rem'}),
                          html.Div([
                              dmc.Select(
                                        #label="На дату",
                                        placeholder="На дату",
                                        id="date_start_dropdown",
                                        value=dates[3],
                                        data=[{'label':val, 'value':val} for val in dates],
                                        radius=20,
                                        style={"width": 80, 
                                               "marginBottom": 10,
                                               "marginTop": 10},
                                    ),
                              ]),
                          html.Div([
                              dmc.Select(
                                        #label="Дата отчета",
                                        placeholder="Дата отчета",
                                        id="date_end_dropdown",
                                        value=dates[-1],
                                        data=[{'label':val, 'value':val} for val in dates],
                                        radius=20,
                                        style={"width": 80, 
                                               "marginBottom": 10,
                                               "marginTop": 10},
                                    ),
                              ],style={"padding-left":'20px'}),                     
                      ],style={"display":"flex",'padding-right':'40px',
                               'align-items':'center'}),
                   ],
               ),
    html.Div([        
        dmc.Grid(gutter="md", children=[
                 
                dmc.Col(
                     dmc.Card(
                         children=[
                             html.Div([
                                 html.Div([], id='avg-days-inplace', className='big-card-value'),
                                 html.Div([
                                     html.Div([
                                         DashIconify(icon="feather:arrow-up-right",
                                                     id='delta-avg-days-icon',
                                                     color="gray", width=20),
                                         html.Div([], id='delta-avg-days', className='card-delta'),
                                     ], className='card-delta-container'),
                                     html.Div(['Дня'], className='card-value-label',),
                                 ],),
                             ], className='big-card-value-container'),
                             html.Div(["""Средняя 
                                        продолжительность 
                                        пребывания граждан 
                                        в коллективных средствах размещения
                                        """], className='card-describe'),                             
                            dcc.Graph(id='avg-days-plot',
                                           config={'displayModeBar':False,
                                                   'scrollZoom':False,
                                                   'doubleClick':False},
                                           className='avg-days-plot',
                                           )
                             ],
                    shadow="sm",
                    style={'height':'25vh', 'width': '100%'},
                    className='tat_card',
                    ),
                    span=2),
                 
                 dmc.Col(
                     dmc.Card(
                         children=[                             
                             html.Div([
                                 html.Div([
                                     html.Div([], id='avg-people-inplace', className='big-card-value'),
                                     html.Div([
                                         html.Div([
                                             DashIconify(icon="feather:arrow-up-right",
                                                         id='delta-avg-people-icon',
                                                         color="gray", width=20),
                                             html.Div([], id='delta-avg-people', className='card-delta'),
                                         ], className='card-delta-container'),
                                         html.Div(['Млн чел.'], className='card-value-label',),
                                     ],),
                                 ], className='big-card-value-container'),
                                 html.Div(["""Численность размещенных
                                            в коллективных средствах
                                            размещения, чел.
                                            """], className='card-describe'),
                                 dmc.Space(h=10),
                                 dbc.Progress(color='#fff', id='people-progress'),
                                 dmc.Space(h=10),
                                 html.Div([
                                     html.Div([
                                         html.Div(['Иностранцы'], id='people-foreign-label'),
                                         html.Div([], id='people-foreign-perc-value'),
                                         html.Div([], id='people-foreign-value'),
                                     ],),
                                     html.Div([
                                         html.Div(['Граждане РФ'], id='people-rf-label'),
                                         html.Div([], id='people-rf-perc-value'),
                                         html.Div([], id='people-rf-value'),
                                     ],style={'display':'grid','justify-items':'end'}),
                                     ], className='people-postprogress-container'),
                                 html.Div([
                                     dcc.Graph(id='people-plot',
                                                    config={'displayModeBar':False,
                                                            'scrollZoom':False,
                                                            'doubleClick':False},
                                                    className='people-plot',)
                                     ], className='people-fig-container'), 
                             ],className='people-main-container'),
                             ],
                    shadow="sm",
                    className='tat_card',
                    style={'height':'25vh', 'width': '100%'},
                    ),
                    span=2),
                 
                 dmc.Col(
                     dmc.Card(
                         children=[ 
                             dmc.Text("Коллективные средства размещения", weight=500),
                             html.Div([
                                 html.Div([
                                     html.Div([
                                         html.Div([], id='profit-inplace', className='medium-card-value'),
                                         html.Div([
                                             html.Div([
                                                 DashIconify(icon="feather:arrow-up-right",
                                                             id='delta-profit-inplace-icon',
                                                             color="gray", width=20),
                                                 html.Div([], id='delta-profit-inplace', className='card-delta'),
                                             ], className='card-delta-container'),
                                             html.Div(['Млн руб.'], className='card-value-label',),
                                         ],),                                
                                     ], className='big-card-value-container'),
                                     html.Div(["""Доходы коллективных
                                               средств размещения
                                               """], className='card-describe'),
                                     html.Div([
                                         dcc.Graph(id='profit-inplace-plot',
                                                        config={'displayModeBar':False,
                                                                'scrollZoom':False,
                                                                'doubleClick':False},
                                                        className='profit-inplace-plot'),
                                         ], className='profit-inplace-fig-container'),                                 
                                 ],),
                                 html.Div([
                                    html.Div([
                                        html.Div([], id='count-inplace', className='medium-card-value'),
                                        html.Div([
                                            html.Div([
                                                DashIconify(icon="feather:arrow-up-right",
                                                            id='delta-count-inplace-icon',
                                                            color="gray", width=20),
                                                html.Div([], id='delta-count-inplace', className='card-delta'),
                                            ], className='card-delta-container'),
                                            html.Div(['ед.'], className='card-value-label',),
                                        ],),                                    
                                    ], className='big-card-value-container'),
                                    html.Div(["""Число коллективных
                                              средств размещения
                                              """], className='card-describe'),
                                    html.Div([
                                        dcc.Graph(id='count-inplace-plot',
                                                       config={'displayModeBar':False,
                                                               'scrollZoom':False,
                                                               'doubleClick':False},
                                                       className='count-inplace-plot'),
                                        ], className='count-inplace-fig-container'),                                 
                                 ], className='inplace-col-container'),
                                 html.Div([
                                     html.Div([
                                         html.Div([], id='count-nights-inplace', className='medium-card-value'),
                                         html.Div([
                                             html.Div([
                                                 DashIconify(icon="feather:arrow-up-right",
                                                             id='delta-count-nights-inplace-icon',
                                                             color="gray", width=20),
                                                 html.Div([], id='delta-count-nights-inplace', className='card-delta'),
                                             ], className='card-delta-container'),
                                             html.Div(['Млн раз'], className='card-value-label',),
                                         ],),                                    
                                     ], className='big-card-value-container'),
                                     html.Div(["""Число ночевок
                                               в коллективных 
                                               средствах размещения 
                                               """], className='card-describe'),
                                     html.Div([
                                         dcc.Graph(id='count-nights-inplace-plot',
                                                        config={'displayModeBar':False,
                                                                'scrollZoom':False,
                                                                'doubleClick':False},
                                                        className='count-nights-inplace-plot'),
                                         ], className='count-nights-fig-container'),                                 
                                 ],className='inplace-col-container'),
                                
                             ], style={'display':'grid', 'grid-template-columns': '33% 33% 33%'}),
                             ],
                    shadow="sm",
                    className='tat_card',
                    style={'height':'25vh', 'width': '100%'},
                    ),
                    span=4),
                 
                 dmc.Col(
                     dmc.Card(
                         children=[ 
                             dmc.Text("Деятельность туристических фирм", weight=500),
                             html.Div([
                                 html.Div([
                                     html.Div([
                                         html.Div([
                                             html.Div([], id='tourfirm-count-all', className='medium-card-value'),
                                             html.Div([
                                                 html.Div([
                                                     DashIconify(icon="feather:arrow-up-right",
                                                                 id='delta-tourfirm-count-icon',
                                                                 color="gray", width=20),
                                                     html.Div([], id='delta-tourfirm-count', className='card-delta'),
                                                 ], className='card-delta-container'),
                                                 html.Div(['Ед.'], className='card-value-label',),
                                             ],),                                   
                                         ], className='big-card-value-container'),
                                         html.Div(["Число турфирм"], className='card-describe'),
                                         html.Div([
                                             html.Div([
                                                 html.Div(['Турагенская'], 
                                                          id='tourfirm-tour-progress-label',
                                                          className='tour-label-progress'),
                                                 dbc.Progress(color='#fff', id='tourfirm-tour-progress'),
                                                 html.Div([], id='tourfirm-tour-progress-abs-value'),
                                                 ], className='tourfirm-progress-container'),
                                             html.Div([
                                                 html.Div(['Туроаператорская'], 
                                                          id='tourfirm-oper-progress-label',
                                                          className='tour-label-progress'),
                                                 dbc.Progress(color='#fff', id='tourfirm-oper-progress'),
                                                 html.Div([], id='tourfirm-oper-progress-abs-value'),
                                                 ], className='tourfirm-progress-container'),
                                             html.Div([
                                                 html.Div(['Туроператорская и турагентская'], 
                                                          id='tourfirm-touroper-progress-label',
                                                          className='tour-label-progress'),
                                                 dbc.Progress(color='#fff', id='tourfirm-touroper-progress'),
                                                 html.Div([], id='tourfirm-touroper-progress-abs-value'),
                                                 ], className='tourfirm-progress-container'),
                                             ], className='tourfirm-progresses-container'),                                 
                                         ],),
                                     ]),
                                 html.Div([
                                     html.Div([
                                         html.Div([
                                             html.Div([], id='tours-count-all', className='medium-card-value'),
                                             html.Div([
                                                 html.Div([
                                                     DashIconify(icon="feather:arrow-up-right",
                                                                 id='delta-tours-count-all-icon',
                                                                 color="gray", width=20),
                                                     html.Div([], id='delta-tours-count-all', className='card-delta'),
                                                 ], className='card-delta-container'),
                                                 html.Div(['Тыс шт.'], className='card-value-label',),
                                             ],),                                    
                                         ], className='big-card-value-container'),
                                         html.Div(["Число турпакетов"], className='card-describe'),
                                         html.Div([
                                             html.Div([
                                                 html.Div(['Гр-н РФ по территории РФ'], 
                                                          id='tours-rfrf-count-progress-label',
                                                          className='tour-label-progress'),
                                                 dbc.Progress(color='#fff', id='tours-rfrf-progress'),
                                                 html.Div([], id='tours-rfrf-progress-abs-value'),
                                                 ], className='tourfirm-progress-container'),
                                             html.Div([
                                                 html.Div(['Гр-н РФ по другим странам'], 
                                                          id='tours-rfother-count-progress-label',
                                                          className='tour-label-progress'),
                                                 dbc.Progress(color='#fff', id='tours-rfother-progress'),
                                                 html.Div([], id='tours-rfother-progress-abs-value'),
                                                 ], className='tourfirm-progress-container'),
                                             html.Div([
                                                 html.Div(['Гр-н других стран по территории РФ'], 
                                                          id='tours-otherrf-count-progress-label',
                                                          className='tour-label-progress'),
                                                 dbc.Progress(color='#fff', id='tours-otherrf-progress'),
                                                 html.Div([], id='tours-otherrf-progress-abs-value'),
                                                 ], className='tourfirm-progress-container'),
                                             ], className='tourfirm-progresses-container'),                                 
                                         ],),
                                     ], style={'border-left': '0.1rem solid',
                                                'border-left-color': 'rgb(255 255 255 / 40%)',
                                                'padding-left': '15px'}),                                 
                             ], className='tour-card-grid-container'),
                         ],
                    shadow="sm",
                    className='tat_card',
                    style={'height':'25vh', 'width': '100%'},
                    ),
                    span=4),
                 ], 
            style={'padding-top':'15px', 'margin-bottom':'8px'}),   
        
    dmc.Grid([
        dmc.Col(
            dmc.Card(
                children=[ 
                    dmc.Text("", weight=500, id='header-cluster-pivot-table'),  
                        html.Div([
                            html.Div([
                                dcc.Graph(id='most-famous-pie-in-the-world',
                                          config={'displayModeBar':False,
                                                  'scrollZoom':False,
                                                  'doubleClick':False},),
                                html.Div([
                                    html.Div([
                                        html.Div(['']),
                                        html.Div(['']),
                                        html.Div(['Кол-во отзывов, шт.']),
                                        html.Div(['% от всех']),
                                        DashIconify(icon="feather:circle", color='rgb(144, 238, 144)'),
                                        html.Div(['Позитивные']),
                                        html.Div([], id='pie-positive-value'),
                                        html.Div([], id='pie-positive-perc'),
                                        DashIconify(icon="feather:circle", color='rgb(220, 220, 220)'),
                                        html.Div(['Нейтральные']),
                                        html.Div([], id='pie-neutral-value'),
                                        html.Div([], id='pie-neutral-perc'),
                                        DashIconify(icon="feather:circle", color='rgb(255, 99, 71)'),
                                        html.Div(['Негативные']),
                                        html.Div([], id='pie-negative-value'),
                                        html.Div([], id='pie-negative-perc'),
                                        ], className='pie-label-container'),
                                    html.Div(['Вывод: '], id='pie-analytic-result'),
                                ]),
                            ],className='pie-container'),

                            dmc.Divider(variant="solid", className='dash-devider'),
                            dag.AgGrid(
                                id='cluster-pivot-table',
                                className='ag-theme-balham-dark',
                                rowData='',
                                #columnSize = 'sizeToFit',
                                defaultColDef={"filter": True,
                                               "cellStyle": {"wordBreak": "normal"},
                                               "wrapText": True,
                                               "autoHeight": True,},
                                dashGridOptions={'rowHeight':20,
                                                 'headerHeight':35,
                                                 'groupHeaderHeight':35,
                                                 'wrapHeaderText': True,
                                                 'autoHeaderHeight':True},
                                style={'height':'100%','width':'100%'}
                                ),
                            ],className='bottom-table-container'),
                        ],
               shadow="sm",
               className='tat_card',
               style={'height':'65vh', 'width': '100%', 
                      },
            ),        
            span=4),
                        
        dmc.Col(
                dmc.Card(
                        children=[
                            html.Div([                                
                                dmc.Text("Отзывов по кластерам", weight=500, id='header-voices'),
                                html.Div([
                                    dmc.SegmentedControl(
                                            id="cluster_type_radio",
                                            value='cluster',
                                            data=[
                                                {"value": 'all', "label": "Всего"},
                                                {"value": 'cluster', "label": "По кластерам"},
                                            ],
                                            radius=20,
                                            size='xs',
                                            className='emoji-radio'
                                        ),
                                    ],id='cluster_type_radio-container', style={'display':'block'}),
                            ], className='header-emoji-radio-container'),
                            html.Div([
                                html.Div([
                                    dcc.Graph(id='voice-by-clusters-plot',
                                             config={'displayModeBar':False,
                                                    'scrollZoom':False,
                                                    'doubleClick':False},
                                             className='voice-by-cluster-plot'),
                                ],className='voice-by-cluster-plot-cont'),
                                html.Div([
                                    dcc.Graph(id='voice-by-tone-plot',
                                             config={'displayModeBar':False,
                                                    'scrollZoom':False,
                                                    'doubleClick':False},
                                             className='voice-by-tone-plot'),
                                ],className='voice-by-tone-plot-cont'),
                                html.Div([
                                    dcc.Graph(id='emoji-by-tone-plot',
                                             config={'displayModeBar':False,
                                                    'scrollZoom':False,
                                                    'doubleClick':False},
                                             className='emoji-by-tone-plot'),
                                ],className='emoji-by-cluster-plot-cont'),
                                html.Div([
                                    html.Div([
                                        dmc.Text("Положительные эмодзи", weight=500,
                                                 id='header-positive-emoji-container'),
                                        html.Div(id='positive-emoji-stuck'),
                                        ], id='positive-emoji-container'),
                                    html.Div([
                                        html.Div([
                                            dmc.Text("Отрицательные эмодзи", weight=500,
                                                     id='header-negative-emoji-container'),                                            
                                        ],className='header-emoji-radio-container'),
                                        html.Div(id='negative-emoji-stuck'),
                                        ], id='negative-emoji-container'),
                                ],className='emoji-tone-container'),
                            ],className='bottom-figs-container'),
                        ],
                       shadow="sm",
                       className='tat_card',
                       style={'height':'65vh', 'width': '100%', 
                              },
                    ),span=4),
        
        dmc.Col(               
            dmc.Card(
                children=[
                    html.Div([
                        dmc.Text("", weight=500),
                        html.Div([
                            dmc.SegmentedControl(
                                    id="famous_type_radio",
                                    value='cluster',
                                    data=[
                                        {"value": 'cluster', "label": "Кластер"},
                                        {"value": 'famous', "label": "Тема"},
                                    ],
                                    radius=20,
                                    size='xs',
                                    className='emoji-radio'
                                ),
                            ], className='famous-cluster-select-container'),
                        dmc.Menu(
                                [
                                    dmc.MenuTarget(dmc.Button("Фильтры", 
                                                              leftIcon=DashIconify(icon="feather:filter"),
                                                              variant="light",
                                                              color='blue',
                                                              radius='xl')),
                                    dmc.MenuDropdown(
                                        [   
                                            dmc.MenuLabel("Дата"),
                                            dmc.MultiSelect(
                                                id='filter-date',
                                                data=[{'value':val, 'label':val} for val in sorted(df_tat['Год_месяц'].unique(), reverse=True)],
                                                value='',
                                                clearable=True,
                                                style={"width": 160},
                                            ),
                                            dmc.MenuDivider(), 
                                            dmc.MenuLabel("Кластер"),
                                            dmc.MultiSelect(
                                                id='filter-cluster',
                                                data=[{'value':val, 'label':val} for val in cluster_describe['Num_Cluster'].unique()],
                                                value='',
                                                clearable=True,
                                                style={"width": 160},
                                            ),
                                            dmc.MenuDivider(), 
                                            dmc.MenuLabel("Тональность отзыва"),
                                            dmc.MultiSelect(
                                                id='filter-tone',
                                                data=[{'value':val, 'label':val} for val in table_tone['Тональность'].unique()],
                                                value='',
                                                clearable=True,
                                                style={"width": 160},
                                            ),
                                            dmc.MenuDivider(), 
                                            dmc.MenuLabel("Тематика"),
                                            dmc.MultiSelect(
                                                id='filter-theme',
                                                data=[{'value':val, 'label':val} for val in sorted(themes_count['Тема'].unique())],
                                                value='',
                                                clearable=True,
                                                searchable=True,
                                                nothingFound="No options found",
                                                style={"width": 160},
                                            )
                                        ]
                                    ),
                                ],
                                position ='left',
                                transition='slide-left',
                                transitionDuration=150,
                            ) 
                    ], className='header-wordcloud-radio-container'),
                    html.Div([
                        html.Div([
                            dcc.Graph(id='wordcloud',
                                     config={'displayModeBar':False,
                                            'scrollZoom':False,
                                            })
                            ],className='wordcloud-container'),
                        html.Div([
                            dag.AgGrid(
                                id='voice-text-table',
                                className='ag-theme-balham-dark',
                                rowData='',
                                columnDefs  = [
                                        {
                                        'children': [
                                            {'field':'index', 'headerName':'№', 'width': 54},
                                            {'field':'Тест', 'headerName':'Отзыв', 'width': 460},
                                            {'field':'Тональность', 'headerName':'Тон-ть', 'width': 80},
                                            {'field':'Emoji', 'headerName':'Эмодзи', 'width': 94},
                                            {'field':'Emoji_tone', 'headerName':'Тон-ть', 'width': 80}, 
                                            ],
                                        },
                                    ],
                                #columnSize = 'sizeToFit',
                                defaultColDef={'filter': True,
                                               'cellStyle': {'wordBreak': 'normal'},
                                               'wrapText': True,
                                               'autoHeight': True,},
                                dashGridOptions={'rowHeight':20,
                                                 'headerHeight':35,
                                                 'groupHeaderHeight':35,
                                                 'wrapHeaderText': True,
                                                 'autoHeaderHeight':True,
                                                 'pagination': True, 
                                                 'animateRows': False,
                                                 'paginationPageSize': 20},
                                style={'height':'100%','width':'100%'}
                                ),
                            ], className='voice-text-table-container')                        
                        ],className='bottom-last-container'),                            
                    ],
                   shadow="sm",
                   className='tat_card',
                   style={'height':'65vh', 'width': '100%', 
                         },
                ),span=4),                        
        ],className='bottom-container'), 
    ], style={'padding-left':'20px','padding-right':'20px'}),
    
],)

#----------------------------END OF THE LAYOUT--------------------------------#
#-----------------------------------------------------------------------------#

if __name__ == '__main__':
    app.run_server(debug=False)
