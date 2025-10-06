import streamlit as st
import pandas as pd
import pymysql
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Tourism Dashboard", layout="wide")

def load_data():
    connection = None
    try:
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='Test@123',
            database='Tourism',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM attraction_visits")
            result = cursor.fetchall()
            column_names = [i[0] for i in cursor.description]
            return pd.DataFrame(result, columns=column_names)
    finally:
        if connection:
            connection.close()

        
df = load_data()

st.title("🏖️ Tourism Dashboard")

year_options = ["Select"] + sorted(df['visit_year'].astype(str).unique(), reverse=True)
region_options = ["Select"] + sorted(df["region"].unique().tolist())
continent_options = ["Select"] + sorted(df["continent"].unique().tolist())
country_options = ["Select"] + sorted(df['country'].unique().tolist())
city_options = ["Select"] + sorted(df["city"].unique().tolist())
attraction_options = ["Select"] + sorted(df["attraction_type"].unique().tolist())
visitMode_options = ["Select"] + sorted(df["visit_mode"].unique().tolist())
visitSeason_options = ["Select"] + sorted(df["visit_season"].unique().tolist())
# Sidebar
section = st.sidebar.radio("📌 Choose Section", ["Dashboard", "Prediction", "Classification", "Recommendation"])

# --------------------------------------- DASHBOARD ---------------------------------------  #
if section == "Dashboard":
    st.subheader("📊 Summary Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Visits", len(df))
    col2.metric("Unique Countries", df['country'].nunique())
    col3.metric("Unique Cities", df['city'].nunique())

    def top_n(field, n=3, exclude_na_str=False):
        series = df[field]
        if exclude_na_str:
            series = series[series.notna() & (series.str.upper() != 'NA')]
        return series.value_counts().nlargest(n)

    with st.expander("Top 3 Overview", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🌍 Top Continents")
            st.dataframe(top_n('continent'))

        with col2:
            st.markdown("### 🗺️ Top Countries")
            st.dataframe(top_n('country'))

        with col3:
            st.markdown("### 🏙️ Top Cities")
            st.dataframe(top_n('city',exclude_na_str=True))

        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown("### ⛅ Top Visit Seasons")
            st.dataframe(top_n('visit_season'))

        with col5:
            st.markdown("### 🚶‍♂️ Top Visit Modes")
            st.dataframe(top_n('visit_mode'))

        with col6:
            st.markdown("### 🌐 Top Regions")
            st.dataframe(top_n('region'))


# ========================== PREDICTION ========================== #
elif section == "Prediction":
    st.subheader("📈 Visit Count Prediction (Linear Regression)")

    pred_year = st.sidebar.number_input("Minimum Year", value=2023, key="year")
    pred_month = st.sidebar.slider("Max Results", 1, 12, 3, key="month")
    pred_region = st.sidebar.selectbox("Region", region_options, key="region")
    pred_continent = st.sidebar.selectbox("Continent", continent_options, key="continent")
    pred_country = st.sidebar.selectbox("Country", country_options, key="country")
    pred_city = st.sidebar.selectbox("City", city_options, key="city")
    pred_attraction = st.sidebar.selectbox("Attraction Type", attraction_options, key="attraction")
    pred_visitmode = st.sidebar.selectbox("Visit Mode", visitMode_options, key="visit_mode")
    pred_visitseason = st.sidebar.selectbox("Visit Season", visitSeason_options, key="visit_season")
    pred_button = st.sidebar.button("Apply")


    features = [
            'visit_year', 'visit_month', 'visit_mode_encoded',
            'visit_season_encoded', 'attraction_type_encoded',
            'continent_encoded', 'region_encoded', 'country_encoded', 'city_encoded'
        ]

    X = df[features]
    y = df['rating'] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    print("Linear Regression RMSE:", root_mean_squared_error(y_test, y_pred_lr))
    print("R2 Score:", r2_score(y_test, y_pred_lr))
    # st.write(f"Linear Regression RMSE: **{root_mean_squared_error(y_test, y_pred_lr)}**")
    # st.write(f"R2 Score : **{r2_score(y_test, y_pred_lr)}**")

    if pred_button:
        
        selected_values = {
            'visit_mode': pred_visitmode,
            'visit_season': pred_visitseason,
            'continent': pred_continent,
            'country': pred_country,
            'region': pred_region,
            'city': pred_city,
            'attraction_type': pred_attraction
        }

        encoded_params = {}

        
        for col, value in selected_values.items():
                encoded_col = f"{col}_encoded"
               
                if value and value != 'Select':
                    filtered = df[df[col] == value]
                    if not filtered.empty and encoded_col in df.columns:
                        encoded_params[col] = filtered[encoded_col].iloc[0]
                    else:
                        encoded_params[col] = 0  
                else:
                    encoded_params[col] = 0  

        new_data = pd.DataFrame([{
            'visit_year': pred_year,
            'visit_month': pred_month,
            'visit_mode_encoded': encoded_params['visit_mode'],               
            'visit_season_encoded': encoded_params['visit_season'],             
            'attraction_type_encoded': encoded_params['attraction_type'] ,        
            'continent_encoded': encoded_params['continent'],
            'region_encoded': encoded_params['region'],
            'country_encoded': encoded_params['country'],
            'city_encoded': encoded_params['city']      
        }])

        predicted_rating = lr.predict(new_data)
        print("Predicted Rating:", predicted_rating[0])

        st.write(f"Predicted Rating : **{predicted_rating[0]:.2f}**")

# ------------------------------------------------------- CLASSIFICATION ------------------------------------------------- #
elif section == "Classification":
    st.subheader("📚 Classification")

    clas_year = st.sidebar.number_input("Minimum Year", value=2023, key="year")
    clas_month = st.sidebar.slider("Max Results", 1, 12, 3, key="month")
    clas_region = st.sidebar.selectbox("Region", region_options, key="region")
    clas_continent = st.sidebar.selectbox("Continent", continent_options, key="continent")
    clas_country = st.sidebar.selectbox("Country", country_options, key="country")
    clas_city = st.sidebar.selectbox("City", city_options, key="city")
    clas_attraction = st.sidebar.selectbox("Attraction Type", attraction_options, key="attraction")
    clas_visitseason = st.sidebar.selectbox("Visit Season", visitSeason_options, key="visit_season")
    clas_button = st.sidebar.button("Apply")


    features = [
    'visit_year', 'visit_month',
    'attraction_type_encoded',
    'continent_encoded', 'country_encoded', 'city_encoded', 'region_encoded',
    'visit_season_encoded'
    ]

    target = 'visit_mode_encoded'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # st.write(f"Confusion Matrix : **{confusion_matrix(y_test, y_pred)}**")
    # st.write(f"Classification Report : **{classification_report(y_test, y_pred)}**")

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    if clas_button:
        selected_values = {              
                'attraction_type': clas_attraction,
                'continent': clas_continent,
                'country': clas_country,
                'city': clas_city,
                'region': clas_region,
                'visit_season' : clas_visitseason
                }

        encoded_params = {}

        for col, value in selected_values.items():
                encoded_col = f"{col}_encoded"

                if value and value != 'Select':
                    filtered = df[df[col] == value]
                    if not filtered.empty and encoded_col in df.columns:
                        encoded_params[col] = filtered[encoded_col].iloc[0]
                    else:
                        encoded_params[col] = 0  
                else:
                    encoded_params[col] = 0 

        new_data = pd.DataFrame([{
            'visit_year': clas_year,
            'visit_month': clas_month,
            'visit_season_encoded': encoded_params['visit_season'],             
            'attraction_type_encoded': encoded_params['attraction_type'] ,         
            'continent_encoded': encoded_params['continent'],
            'region_encoded': encoded_params['region'],
            'country_encoded': encoded_params['country'],
            'city_encoded': encoded_params['city']      
        }])

        new_input_df = new_data[features]
        predicted_class = clf.predict(new_input_df)[0]
        print("Predicted Visit Mode Encoded:", predicted_class)

        visit_mode_reverse_map = {1: 'Business', 2: 'Couples', 3: 'Family', 4: 'Friends'}
        print("Predicted Visit Mode:", visit_mode_reverse_map[predicted_class])

    

        st.text("📋 Classification Report:")
        st.write(f"Predicted Visit Mode : **{visit_mode_reverse_map[predicted_class]}**")
        # st.text(classification_report(y_test, y_pred))

# ------------------------------------------------------- RECOMMENDATION ------------------------------------------------------- #
elif section == "Recommendation":
    st.subheader("🧭 Travel Recommendations")

    rec_continent = st.selectbox("Continent", ["All"] + sorted(df['continent'].dropna().unique().tolist()))
    rec_region = st.selectbox("Region", ["All"] + sorted(df['region'].dropna().unique().tolist()))
    rec_visit = st.selectbox("Visit Mode", ["All"] + sorted(df['visit_mode'].dropna().unique().tolist()))
    rec_btn = st.button("Apply")

    # Use 'user_id' if exists, or create synthetic user ID
    df['user'] = df['continent'] + '_' + df['region'] + '_' + df['visit_mode']

    # Pivot table to get user-item matrix
    user_item_matrix = df.pivot_table(index='user', columns='attraction', values='rating').fillna(0)

    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    def recommend_collaborative(target_user, top_n=3):
        if target_user not in user_item_matrix.index:
            raise ValueError(f"User '{target_user}' not found in the dataset.")
        
        sim_users = user_similarity_df[target_user].drop(target_user).sort_values(ascending=False)

            # Only keep users present in the user_item_matrix
        sim_users = sim_users[sim_users.index.isin(user_item_matrix.index)]
            
            # Align indices
        aligned_users = user_item_matrix.loc[sim_users.index]

            # Weighted sum of ratings
        weighted_ratings = aligned_users.T.dot(sim_users)
        summed_weights = sim_users.sum()
            
        if summed_weights == 0:
            raise ValueError("No similar users found to make recommendations.")
            
        recommendations = weighted_ratings / summed_weights
        # Remove already rated attractions
        already_rated = user_item_matrix.loc[target_user]
        recommendations = recommendations[already_rated == 0]

        return recommendations.sort_values(ascending=False).head(top_n)
   
    if rec_btn:
        try:
            st.markdown("### 🌟 Top Recommended Places")

            recommendations = recommend_collaborative(rec_continent + "_" + rec_region + "_" + rec_visit)

            # Convert recommended attraction names into a DataFrame
            top_attractions = recommendations.index.tolist()
            rec_df = df[['attraction', 'attraction_address']].drop_duplicates()
            rec_details = rec_df[rec_df['attraction'].isin(top_attractions)]

            st.dataframe(rec_details[['attraction', 'attraction_address']].reset_index(drop=True))

        except ValueError as e:
            st.warning(str(e))

