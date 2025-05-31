import streamlit as st
import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import date

st.title("Local Food Wastage Management System")

# Use persistent SQLite DB file
db_file = "food_waste.db"

# Initialize database only once
@st.cache_resource
def create_connection():
    conn = sqlite3.connect(db_file, check_same_thread=False)
    return conn

conn = create_connection()
cursor = conn.cursor()

# Load CSV data only if tables don't exist (initial setup)
def initialize_database():
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    if "providers" not in tables:
        pd.read_csv("providers_data.csv").to_sql("providers", conn, index=False)
    if "receivers" not in tables:
        pd.read_csv("receivers_data.csv").to_sql("receivers", conn, index=False)
    if "food_listings" not in tables:
        pd.read_csv("food_listings_data.csv").to_sql("food_listings", conn, index=False)
    if "claims" not in tables:
        pd.read_csv("claims_data.csv").to_sql("claims", conn, index=False)

initialize_database()

# Export database tables to CSV
if st.sidebar.button("Export Tables to CSV"):
    try:
        pd.read_sql_query("SELECT * FROM providers", conn).to_csv("export_providers.csv", index=False)
        pd.read_sql_query("SELECT * FROM receivers", conn).to_csv("export_receivers.csv", index=False)
        pd.read_sql_query("SELECT * FROM food_listings", conn).to_csv("export_food_listings.csv", index=False)
        pd.read_sql_query("SELECT * FROM claims", conn).to_csv("export_claims.csv", index=False)
        st.sidebar.success("Exported to CSV files successfully.")
    except Exception as e:
        st.sidebar.error(f"Export failed: {e}")

section = st.sidebar.radio("Navigate", ["Providers", "Receivers", "Food Listings", "Claims", "Update Listing", "Add Listing", "Delete Listing", "Queries"])

if section == "Providers":
    st.header("Food Providers")
    df = pd.read_sql_query("SELECT * FROM providers", conn)
    st.dataframe(df)

elif section == "Receivers":
    st.header("Food Receivers")
    df = pd.read_sql_query("SELECT * FROM receivers", conn)
    st.dataframe(df)

elif section == "Food Listings":
    st.header("Food Listings")
    df = pd.read_sql_query("SELECT * FROM food_listings", conn)
    st.dataframe(df)

elif section == "Claims":
    st.header("Food Claims")
    df = pd.read_sql_query("SELECT * FROM claims", conn)
    st.dataframe(df)

elif section == "Update Listing":
    st.header("Update Food Listing")
    listings = pd.read_sql_query("SELECT * FROM food_listings", conn)

    if listings.empty:
        st.info("No listings to update.")
    else:
        food_id = st.selectbox("Select Listing by ID", listings["Food_ID"])
        listing = listings[listings["Food_ID"] == food_id].iloc[0]

        new_name = st.text_input("Food Name", value=listing["Food_Name"])
        new_qty = st.number_input("Quantity", value=int(listing["Quantity"]), min_value=1)
        new_expiry = st.date_input("Expiry Date", value=pd.to_datetime(listing["Expiry_Date"]).date() if pd.notnull(listing["Expiry_Date"]) else date.today())

        if st.button("Update Listing"):
            try:
                cursor.execute("""
                    UPDATE food_listings
                    SET Food_Name = ?, Quantity = ?, Expiry_Date = ?
                    WHERE Food_ID = ?
                """, (new_name, new_qty, new_expiry.isoformat(), food_id))
                conn.commit()
                st.success("Listing updated.")
            except Exception as e:
                st.error(f"Error: {e}")

elif section == "Add Listing":
    st.header("Add a New Food Listing")

    provider_id = st.selectbox("Select Provider ID", pd.read_sql_query("SELECT Provider_ID FROM providers", conn)["Provider_ID"])
    provider_info = pd.read_sql_query(f"SELECT Type, Location FROM providers WHERE Provider_ID = {provider_id}", conn).iloc[0]
    st.markdown(f"**Provider Type:** {provider_info['Type']}  ")
    st.markdown(f"**Location:** {provider_info['Location']}")

    food_name = st.text_input("Food Name")
    food_type = st.selectbox("Food Type", ["Vegetables", "Fruits", "Grains", "Dairy", "Beverages", "Others"])
    meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snacks"])
    quantity = st.number_input("Quantity", min_value=1)
    expiry_date = st.date_input("Expiry Date")

    if st.button("Add Listing"):
        try:
            cursor.execute("""
                INSERT INTO food_listings (Provider_ID, Food_Name, Food_Type, Meal_Type, Quantity, Expiry_Date, Location)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (provider_id, food_name, food_type, meal_type, quantity, expiry_date.isoformat(), provider_info['Location']))
            conn.commit()
            st.success("Listing added successfully.")
        except Exception as e:
            st.error(f"Error: {e}")

elif section == "Delete Listing":
    st.header("Delete a Food Listing")
    df = pd.read_sql_query("SELECT Food_ID, Food_Name FROM food_listings", conn)

    if not df.empty:
        selected_id = st.selectbox("Select Listing to Delete", df["Food_ID"])
        selected_name = df[df["Food_ID"] == selected_id]["Food_Name"].values[0]
        st.write(f"Selected: {selected_name} (ID: {selected_id})")

        if st.button("Delete Listing"):
            try:
                cursor.execute("DELETE FROM food_listings WHERE Food_ID = ?", (selected_id,))
                conn.commit()
                st.success("Listing deleted successfully.")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("No listings available.")

elif section == "Queries":
    st.header("SQL Queries")

    queries = {
        "1. Providers by City": {
            "sql": "SELECT city, COUNT(*) AS total_providers FROM providers GROUP BY city;",
            "chart": "bar"
        },
        "2. Receivers by City": {
            "sql": "SELECT city, COUNT(*) AS total_receivers FROM receivers GROUP BY city;",
            "chart": "bar"
        },
        "3. Most Active Providers": {
            "sql": "SELECT provider_id, COUNT(*) AS listings FROM food_listings GROUP BY provider_id ORDER BY listings DESC LIMIT 5;",
            "chart": "bar"
        },
        "4. Most Active Claimers": {
            "sql": "SELECT receiver_id, COUNT(*) AS claims FROM claims GROUP BY receiver_id ORDER BY claims DESC LIMIT 5;",
            "chart": "bar"
        },
        "5. Top Claimed Food Types": {
            "sql": "SELECT food_listings.Food_Type, COUNT(*) AS total_claims FROM claims JOIN food_listings ON claims.Food_ID = food_listings.Food_ID GROUP BY food_listings.Food_Type ORDER BY total_claims DESC LIMIT 5;",
            "chart": "bar"
        },
        "6. Listings by Food Type": {
            "sql": "SELECT food_type, COUNT(*) AS count FROM food_listings GROUP BY food_type;",
            "chart": "pie"
        },
        "7. Percentage of Completed Claims": {
            "sql": "SELECT ROUND(100.0 * SUM(CASE WHEN status='Completed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS completion_rate FROM claims;",
            "chart": None
        },
        "8. Average Time to Claim (in Days)": {
            "sql": "SELECT ROUND(AVG(JULIANDAY(claims.Timestamp) - JULIANDAY(food_listings.Expiry_Date)), 2) AS avg_days FROM claims JOIN food_listings ON claims.Food_ID = food_listings.Food_ID WHERE claims.Timestamp IS NOT NULL AND food_listings.Expiry_Date IS NOT NULL;",
            "chart": None
        },
        "9. Listings without Any Claims": {
            "sql": "SELECT COUNT(*) AS unclaimed_listings FROM food_listings WHERE Food_id NOT IN (SELECT Food_id FROM claims);",
            "chart": None
        },
        "10. City with Highest Demand": {
            "sql": "SELECT food_listings.location, COUNT(*) AS total_claims FROM claims JOIN food_listings ON claims.Food_id = food_listings.Food_id GROUP BY food_listings.location ORDER BY total_claims DESC LIMIT 1;",
            "chart": None
        },
        "11. Daily Claim Trends": {
            "sql": "SELECT DATE(Timestamp) AS claim_date, COUNT(*) AS total_claims FROM claims GROUP BY claim_date ORDER BY claim_date;",
            "chart": "line"
        },
        "12. Most Common Meal Types": {
            "sql": "SELECT Meal_Type, COUNT(*) AS count FROM food_listings GROUP BY Meal_Type ORDER BY count DESC;",
            "chart": "bar"
        },
        "13. Claims by Provider Type": {
            "sql": "SELECT providers.Type, COUNT(*) AS total_claims FROM claims JOIN food_listings ON claims.Food_ID = food_listings.Food_ID JOIN providers ON food_listings.Provider_ID = providers.Provider_ID GROUP BY providers.Type;",
            "chart": "bar"
        },
        "14. Average Quantity per Listing": {
            "sql": "SELECT ROUND(AVG(Quantity), 2) AS avg_quantity FROM food_listings;",
            "chart": None
        },
        "15. Top 5 Cities by Listings": {
            "sql": "SELECT city, COUNT(*) AS total_listings FROM providers JOIN food_listings ON providers.Provider_ID = food_listings.Provider_ID GROUP BY city ORDER BY total_listings DESC LIMIT 5;",
            "chart": "bar"
        },
        "16. Expired Listings": {
            "sql": "SELECT COUNT(*) AS expired_listings FROM food_listings WHERE DATE(Expiry_Date) < DATE('now');",
            "chart": None
        },
        "17. Claim Completion Status Breakdown": {
            "sql": "SELECT Status, COUNT(*) AS count FROM claims GROUP BY Status;",
            "chart": "pie"
        }

    }

    selected_query = st.selectbox("Choose a query", list(queries.keys()))

    if st.button("Run Query"):
        q = queries[selected_query]
        df = pd.read_sql_query(q["sql"], conn)
        st.dataframe(df)

        if q["chart"] == "bar" and df.shape[1] >= 2:
            fig, ax = plt.subplots()
            sns.barplot(x=df.columns[0], y=df.columns[1], data=df, ax=ax)
            ax.set_title(selected_query)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        elif q["chart"] == "pie" and df.shape[1] >= 2:
            fig, ax = plt.subplots()
            ax.pie(df.iloc[:, 1], labels=df.iloc[:, 0], autopct='%1.1f%%', startangle=90)
            ax.axis("equal")
            st.pyplot(fig)
