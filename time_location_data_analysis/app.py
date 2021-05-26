import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly.express as px
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium.plugins import FastMarkerCluster
from datetime import date


app = dash.Dash(__name__)


def get_data():
    # read the data required for this app
    d = pd.read_csv('Police_Department_Incident_Reports__2018_to_Present.csv')
    columns_to_keep = [
        'Incident Datetime',
        'Incident Date',
        'Incident Time',
        'Incident Year',
        'Incident Day of Week',
        'Row ID',
        'Incident ID',
        'Incident Code',
        'Incident Category',
        'Incident Subcategory',
        'Incident Description',
        'Police District',
        'Analysis Neighborhood',
        'Latitude',
        'Longitude',
    ]
    d = d[columns_to_keep]
    d.columns = [c.replace(' ', '_').lower() for c in d.columns]
    d.dropna(axis=0, inplace=True)
    d['incident_datetime'] = pd.to_datetime(d['incident_datetime'])

    # read the geo JSON
    df = gpd.read_file('Current Police Districts.geojson')

    return d, df


def update_data(d, df, start_date=None, end_date=None):

    # apply date filters
    if start_date is not None and end_date is not None:
        d = d[((d['incident_datetime'] >= start_date) & (d['incident_datetime'] <= end_date))]

    # create grouping of the data to show crime counts by category and district
    crime_by_catg = d.groupby('incident_category')['row_id'].count().reset_index().rename(columns={"row_id": "count"})
    crime_by_district = d.groupby('police_district')['row_id'].count().reset_index().rename(columns={"row_id": "count"})

    # plotly figures for these groupings
    fig_crime_by_catg = px.bar(crime_by_catg, x='incident_category', y='count')
    fig_crime_by_district = px.bar(crime_by_district, x='police_district', y='count')

    # set up day and hour variables
    d['incident_day'] = d['incident_datetime'].dt.day_name()
    d['incident_hour'] = d['incident_datetime'].dt.hour.astype('category')

    # create groupings of the data to show crime percentages by category and hour, and category and day
    crime_by_catg_and_day = d.groupby(['incident_category', 'incident_day'])['row_id'].count().reset_index().rename(columns={"row_id": "count"})
    crime_by_catg_and_day['perc'] = crime_by_catg_and_day.groupby('incident_category')['count'].apply(lambda x: x / x.sum()).fillna(0)
    crime_by_catg_and_hour = d.groupby(['incident_category', 'incident_hour'])['row_id'].count().reset_index().rename(columns={"row_id": "count"})
    crime_by_catg_and_hour['perc'] = crime_by_catg_and_hour.groupby('incident_category')['count'].apply(lambda x: x / x.sum()).fillna(0)

    fig_crime_by_catg_and_day = px.bar(crime_by_catg_and_day, x='incident_category', y='perc', color='incident_day', barmode='stack')
    fig_crime_by_catg_and_hour = px.bar(crime_by_catg_and_hour, x='incident_category', y='perc', color='incident_hour', barmode='stack')

    # create groupings of the data to show crime percentages by district and hour, and district and day
    crime_by_district_and_day = d.groupby(['police_district', 'incident_day'])['row_id'].count().reset_index().rename(columns={"row_id": "count"})
    crime_by_district_and_day['perc'] = crime_by_district_and_day.groupby('police_district')['count'].apply(lambda x: x / x.sum()).fillna(0)
    crime_by_district_and_hour = d.groupby(['police_district', 'incident_hour'])['row_id'].count().reset_index().rename(columns={"row_id": "count"})
    crime_by_district_and_hour['perc'] = crime_by_district_and_hour.groupby('police_district')['count'].apply(lambda x: x / x.sum()).fillna(0)

    fig_crime_by_district_and_day = px.bar(crime_by_district_and_day, x='police_district', y='perc', color='incident_day', barmode='stack')
    fig_crime_by_district_and_hour = px.bar(crime_by_district_and_hour, x='police_district', y='perc', color='incident_hour', barmode='stack')

    return {
        'fig_crime_by_catg': fig_crime_by_catg,
        'fig_crime_by_district': fig_crime_by_district,
        'fig_crime_by_catg_and_day': fig_crime_by_catg_and_day,
        'fig_crime_by_catg_and_hour': fig_crime_by_catg_and_hour,
        'fig_crime_by_district_and_day': fig_crime_by_district_and_day,
        'fig_crime_by_district_and_hour': fig_crime_by_district_and_hour,
    }


def update_maps(d, df, start_date=None, end_date=None):

    # apply date filters
    if start_date is not None and end_date is not None:
        d_first_map = d[((d['incident_datetime'] >= start_date) & (d['incident_datetime'] <= end_date))]
    else:
        d_first_map = d

    # folium map showing the locations of every crime, clustered depending on zoom level
    fmap = folium.Map(
        location=[d_first_map.latitude.mean(), d_first_map.longitude.mean()], zoom_start=13
    )
    fmap.add_child(
        FastMarkerCluster(
            d_first_map[['latitude', 'longitude']].values.tolist()
        )
    )
    folium.GeoJson(df['geometry']).add_to(fmap)
    fmap.save('map.html')

    # folium map showing the crime loci by police district and year
    district_centroid_by_year = d[d['police_district'] != 'Out of SF'].groupby(['police_district', 'incident_year'])[
        ['latitude', 'longitude']].mean().reset_index(drop=False)
    ffmap = folium.Map(
        location=[d.latitude.mean(), d.longitude.mean()], zoom_start=13
    )
    folium.GeoJson(df['geometry']).add_to(ffmap)

    yrs = list(set(district_centroid_by_year['incident_year']))
    nbr_yrs = len(yrs)
    yr_opacity = np.linspace(0, 1, nbr_yrs)
    opacity_map = dict(zip(yrs, yr_opacity))

    coords = zip(
        district_centroid_by_year['police_district'],
        district_centroid_by_year['incident_year'],
        district_centroid_by_year['latitude'],
        district_centroid_by_year['longitude']
    )
    for ci, c in enumerate(coords):
        folium.CircleMarker(
            location=[c[2], c[3]],
            popup=f"{c[0]}, {c[1]}",
            tooltip=f"{c[0]}, {c[1]}",
            color="red",
            fill=True,
            fill_opacity=opacity_map[c[1]]
        ).add_to(ffmap)

    ffmap.save('map2.html')

    # folium map showing the crime loci by category and year
    ctype_centroid_by_year = d[d['police_district'] != 'Out of SF'].groupby(['incident_category', 'incident_year'])[
        ['latitude', 'longitude']].mean().reset_index(drop=False)
    fffmap = folium.Map(
        location=[d.latitude.mean(), d.longitude.mean()], zoom_start=13
    )
    folium.GeoJson(df['geometry']).add_to(fffmap)

    coords = zip(
        ctype_centroid_by_year['incident_category'],
        ctype_centroid_by_year['incident_year'],
        ctype_centroid_by_year['latitude'],
        ctype_centroid_by_year['longitude']
    )
    for ci, c in enumerate(coords):
        folium.CircleMarker(
            location=[c[2], c[3]],
            popup=f"{c[0]}, {c[1]}",
            tooltip=f"{c[0]}, {c[1]}",
            color="red",
            fill=True,
            fill_opacity=opacity_map[c[1]]
        ).add_to(fffmap)

    fffmap.save('map3.html')


# initialize data with no date restrictions
d, df = get_data()
update_maps(d, df)
initial_figs = update_data(d, df)


# build the dash layout
app.layout = html.Div(children=[

    html.H1(children='San Francisco Crime Dashboard'),

    html.Div(
        children="Crimes in San Francisco from 2018 to present."
    ),

    html.Br(),

    html.P("Choose a date range"),

    dcc.DatePickerRange(
        id='crime_date_range',
        min_date_allowed=date(2018, 1, 1),
        max_date_allowed=date(2021, 3, 1),
        start_date=date(2018, 1, 1),
        end_date=date(2021, 3, 1),
    ),

    dcc.Graph(
        id='crime_by_catg',
        figure=initial_figs['fig_crime_by_catg']
    ),

    dcc.Graph(
        id='crime_by_district',
        figure=initial_figs['fig_crime_by_district']
    ),

    dcc.Graph(
        id='crime_by_catg_and_day',
        figure=initial_figs['fig_crime_by_catg_and_day']
    ),

    dcc.Graph(
        id='crime_by_catg_and_hour',
        figure=initial_figs['fig_crime_by_catg_and_hour']
    ),

    dcc.Graph(
        id='crime_by_district_and_day',
        figure=initial_figs['fig_crime_by_district_and_day']
    ),

    dcc.Graph(
        id='crime_by_district_and_hour',
        figure=initial_figs['fig_crime_by_district_and_hour']
    ),

    html.Br(),

    html.Iframe(
        id='map',
        srcDoc=open('map.html', 'r').read(),
        width='100%',
        height='700'
    ),

    html.Br(),

    html.Iframe(
        id='map_centroid_by_district',
        srcDoc=open('map2.html', 'r').read(),
        width='100%',
        height='700'
    ),

    html.Br(),

    html.Iframe(
        id='map_centroid_by_category',
        srcDoc=open('map3.html', 'r').read(),
        width='100%',
        height='700'
    ),

])


@app.callback(
    [
        Output(component_id="crime_by_catg", component_property="figure"),
        Output(component_id="crime_by_district", component_property="figure"),
        Output(component_id="crime_by_catg_and_day", component_property="figure"),
        Output(component_id="crime_by_catg_and_hour", component_property="figure"),
        Output(component_id="crime_by_district_and_day", component_property="figure"),
        Output(component_id="crime_by_district_and_hour", component_property="figure"),
        Output(component_id="map", component_property="srcDoc"),
        Output(component_id="map_centroid_by_district", component_property="srcDoc"),
        Output(component_id="map_centroid_by_category", component_property="srcDoc"),
    ],
    [
        Input(component_id="crime_date_range", component_property="start_date"),
        Input(component_id="crime_date_range", component_property="end_date"),
    ]
)
def update_figs(start_date, end_date):
    updated_figs = update_data(d, df, start_date, end_date)
    update_maps(d, df, start_date, end_date)
    return (
        updated_figs['fig_crime_by_catg'],
        updated_figs['fig_crime_by_district'],
        updated_figs['fig_crime_by_catg_and_day'],
        updated_figs['fig_crime_by_catg_and_hour'],
        updated_figs['fig_crime_by_district_and_day'],
        updated_figs['fig_crime_by_district_and_hour'],
        open('map.html', 'r').read(),
        open('map2.html', 'r').read(),
        open('map3.html', 'r').read(),
    )


if __name__ == '__main__':
    app.run_server(debug=True)
