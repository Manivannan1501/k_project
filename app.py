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

    provider_id = st.number_input("Provider ID", min_value=1)
    food_name = st.text_input("Food Name")
    quantity = st.number_input("Quantity", min_value=1)
    expiry_date = st.date_input("Expiry Date")

    if st.button("Add Listing"):
        try:
            cursor.execute("""
                INSERT INTO food_listings (Provider_ID, Food_Name, Quantity, Expiry_Date)
                VALUES (?, ?, ?, ?)
            """, (provider_id, food_name, quantity, expiry_date.isoformat()))
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
        "5. Listings by Food Type": {
            "sql": "SELECT food_type, COUNT(*) AS count FROM food_listings GROUP BY food_type;",
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
