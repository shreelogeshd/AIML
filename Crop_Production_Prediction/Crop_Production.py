import streamlit as st
import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

st.set_page_config(page_title="Crop Production Prediction", layout="wide")

try:
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='Test@123',
        database='Crop_Production',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    try:
        mycursor = connection.cursor()

        def fetch_dataframe(query):
            try:
                mycursor.execute(query)
                result = mycursor.fetchall()
                column_names = [i[0] for i in mycursor.description]
                return pd.DataFrame(result, columns=column_names)
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return pd.DataFrame()

        df = fetch_dataframe("SELECT * FROM production")

    finally:
        connection.close()

except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()


area_options = ["Select"] + df['area'].dropna().unique().tolist() if not df.empty else ["Select"]
year_options = ["Select"] + df["year"].dropna().unique().tolist() if not df.empty else ["Select"]
item_options = ["Select"] + df["item"].dropna().unique().tolist() if not df.empty else ["Select"]

st.title("Crop Production 🌾")
st.sidebar.title("Filters")
country = st.sidebar.selectbox("Country", area_options)
year = st.sidebar.selectbox("Year", year_options)
item = st.sidebar.selectbox("Item", item_options)
filter_button = st.sidebar.button("Apply")

# Stats
st.subheader("Crops and livestock products statistics ")


col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Items", df['item_code'].nunique())
with col2:
    st.metric("Countries Harvested", df['area'].nunique())


st.subheader("Production")
try:
    top_countries = df.groupby('area', as_index=False)['production_k'].sum()
    top_countries_sorted = top_countries.sort_values(by='production_k', ascending=False).head(3)
    top_countries_sorted = top_countries_sorted.reset_index(drop=True)
    top_countries_sorted = top_countries_sorted.rename(columns={
        'area': 'Country',
        'production_k': 'Total Production (k tonnes)'
    })
    st.subheader("Top 3 Countries")
    st.dataframe(top_countries_sorted, use_container_width=True)


    top_years = df.groupby('year', as_index=False)['production_k'].sum()
    top_years_sorted = top_years.sort_values(by='production_k', ascending=False).head(3)
    top_years_sorted = top_years_sorted.reset_index(drop=True)

    top_years_sorted = top_years_sorted.rename(columns={
        'year': 'Year',
        'production_k': 'Total Production (k tonnes)'
    })
    top_years_sorted['Year'] = top_years_sorted['Year'].astype(int).astype(str)
    st.subheader("Top 3 Years")
    st.dataframe(top_years_sorted, use_container_width=True)

except Exception as e:
    st.warning(f"Error generating leaderboard: {e}")

try:
    X = df[['area_harvested_k', 'yield_k']]
    y = df['production_k']

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate_model(name, model, X_test, y_test):
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        # rmse = np.sqrt(mse)
        # rmsle = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred)))
        # print(f"{name} Performance:")
        # print(f"R² Score: {r2:.4f}")
        # print(f"Mean Squared Error (MSE): {mse:.4f}")
        # print(f"Mean Absolute Error (MAE): {mae:.4f}")
        # print("-" * 40)
        return {"Model": name, "R2": r2, "MSE": mse, "MAE": mae}
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_metrics = evaluate_model("Linear Regression", lr, X_test, y_test)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model("Random Forest", rf, X_test, y_test)

    results_df = pd.DataFrame([lr_metrics, rf_metrics])
    st.subheader("Evaluation Metrics - Regression")
    st.dataframe(results_df, use_container_width=True)

except Exception as e:
        st.warning(f"Error generating dashboard: {e}")
    
    
if filter_button:
    try:
        
        if (country.lower() != "select" and year != "select" and item.lower() != "select"):
            filtered_df = df[(df['area'] == country) & (df['item'] == item) & (df['year'] == year)]
        elif(country.lower() == "select" and year == "Select" and item.lower() == "select"):
            filtered_df = df[['area', 'item','year','area_harvested_k','production_k','yield_k']].copy()
        elif ((country.lower() != "select") and (year != "select") and (item.lower() == "select")):
            filtered_df = df[(df['area'] == country) & (df['year'] == year)]
        elif ((country.lower() != "select") and (item.lower() != "select") and (year == "select")):
            filtered_df = df[(df['area'] == country) & (df['item'] == item)]
        elif ((year != "select") and (item.lower() != "select") and (country.lower() == "select")):
            filtered_df = df[(df['year'] == year) & (df['item'] == item)]
        elif (country.lower() != "select" and year == "select" and item.lower() == "select"):
            filtered_df = df[df['country'] == country]
        elif (year != "select" and country.lower() == "select" and item.lower() == "select"):
            filtered_df = df[df['year'] == year]        
        elif (item.lower() != "select" and year == "select" and country.lower() == "select"):
            filtered_df = df[df['item'] == item]
        
        selected_columns_df = filtered_df.filter(items=['area', 'item','year','area_harvested_k','production_k','yield_k'])
        filtered_df = selected_columns_df.reset_index(drop=True)
        filtered_df.columns = ['Country','Item','Year','Area Harvested','Production','Yield']
        filtered_df['Year'] = filtered_df['Year'].astype(int).astype(str)

        st.dataframe(filtered_df, use_container_width=True)

    except Exception as e:
        st.warning(f"Error generating dashboard: {e}")



