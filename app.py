from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)

#-----------------------------STATE WISE DATA-----------------------------------

# Load data
df = pd.read_csv("data.csv")

# Prepare data for clustering
state_crimes = df.groupby('state_name')['total_cyber_crimes'].sum().reset_index()
X = state_crimes[['total_cyber_crimes']].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform KMeans clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
state_crimes['cluster'] = kmeans.fit_predict(X_scaled)

# Pass the data to the frontend
states = state_crimes['state_name'].tolist()
clusters = state_crimes['cluster'].tolist()
total_crimes = state_crimes['total_cyber_crimes'].tolist()

#-----------------------------DISTRICT WISE DATA-----------------------------------
crime_columns = [
    'cheating',
    'computer_related_offences',
    'cyber_blackmailing_threatening',
    'fake_news_on_social_media',
    'fraud'
]

# Sum of each crime type
crime_sums = df[crime_columns].sum().tolist()


hide_element = True
@app.route('/')
def index():
    return render_template('country.html', states=states, clusters=clusters, total_crimes=total_crimes, 
                           districts=[], clusters_d=[], total_crimes_d=[], 
                           crime_sums=crime_sums, crime_labels=crime_columns)

@app.route('/state')
def state():
    return render_template('state.html', states=states, clusters=clusters, total_crimes=total_crimes, 
                           districts=[], clusters_d=[], total_crimes_d=[], 
                           crime_sums=crime_sums, crime_labels=crime_columns, hide_element=hide_element)

@app.route('/select_state')
def select_state():
    selected_state = request.args.get('state')

    if selected_state:
        df_s = df[df['state_name'] == selected_state]

        # Prepare data for clustering
        district_crimes = df_s.groupby('district_name')['total_cyber_crimes'].sum().reset_index()
        X_d = district_crimes[['total_cyber_crimes']].values

        # Standardize the data
        scaler_d = StandardScaler()
        X_scaled_d = scaler_d.fit_transform(X_d)

        if len(X_scaled_d) == 1:
            n_clusters_d = 1
        elif len(X_scaled_d) == 2:
            n_clusters_d = 2
        else:
            n_clusters_d = 3
        kmeans_d = KMeans(n_clusters=n_clusters_d, random_state=0)
        district_crimes['cluster'] = kmeans_d.fit_predict(X_scaled_d)
        # Pass the data to the frontend
        districts = district_crimes['district_name'].tolist()
        clusters_d = district_crimes['cluster'].tolist()
        total_crimes_d = district_crimes['total_cyber_crimes'].tolist()

        crime_columns = [
            'cheating',
            'computer_related_offences',
            'cyber_blackmailing_threatening',
            'fake_news_on_social_media',
            'fraud'
        ]

        crime_sums = df_s[crime_columns].sum().tolist()
        hide_element = False
        return render_template('state.html', states=states, clusters=clusters, total_crimes=total_crimes, districts=districts, clusters_d=clusters_d, total_crimes_d=total_crimes_d, crime_sums=crime_sums, crime_labels=crime_columns, hide_element=hide_element)
    return render_template('state.html', states=states, clusters=clusters, total_crimes=total_crimes, districts=[], clusters_d=[], total_crimes_d=[], crime_sums=crime_sums, crime_labels=crime_columns, hide_element=hide_element)

if __name__ == '__main__':
    app.run(debug=True)
