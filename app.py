
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

if section == "Add Listing":
    st.header("Add a New Food Listing")

    provider_id = st.selectbox("Select Provider ID", pd.read_sql_query("SELECT Provider_ID FROM providers", conn)["Provider_ID"])
    provider_info_df = pd.read_sql_query("SELECT Type, Location FROM providers WHERE Provider_ID = ?", conn, params=(provider_id,))
    if provider_info_df.empty:
        st.warning("Provider details not found.")
        st.stop()
    provider_info = provider_info_df.iloc[0]

    st.markdown(f"**Provider Type:** {provider_info['Type']}  ")
    st.markdown(f"**Location:** {provider_info['Location']}")

    food_name = st.text_input("Food Name")
    food_type = st.selectbox("Food Type", ["Vegetables", "Fruits", "Grains", "Dairy", "Beverages", "Others"])
    meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snacks"])
    quantity = st.number_input("Quantity", min_value=1)
    expiry_date = st.date_input("Expiry Date")

    if st.button("Add Listing"):
        try:
            cursor.execute(
                "INSERT INTO food_listings (Provider_ID, Food_Name, Food_Type, Meal_Type, Quantity, Expiry_Date, Location) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (provider_id, food_name, food_type, meal_type, quantity, expiry_date.isoformat(), provider_info['Location'])
            )
            conn.commit()
            st.success("Listing added successfully.")
        except Exception as e:
            st.error(f"Error: {e}")

elif section == "Delete Listing":
    st.header("Delete a Food Listing")
    df = pd.read_sql_query("SELECT Food_ID, Food_Name FROM food_listings", conn)

    if not df.empty:
        selected_id = st.selectbox("Select Listing to Delete", df["Food_ID"])
        selected_row = df[df["Food_ID"] == selected_id]
        if not selected_row.empty:
            selected_name = selected_row["Food_Name"].values[0]
            st.write(f"Selected: {selected_name} (ID: {selected_id})")
        else:
            st.warning("Selected ID not found.")

        if st.button("Delete Listing"):
            try:
                cursor.execute("DELETE FROM food_listings WHERE Food_ID = ?", (selected_id,))
                conn.commit()
                st.success("Listing deleted successfully.")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("No listings available.")
