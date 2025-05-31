
import streamlit as st
import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Local Food Wastage Management System")

def load_data():
    providers = pd.read_csv("providers_data.csv")
    receivers = pd.read_csv("receivers_data.csv")
    food_listings = pd.read_csv("food_listings_data.csv")
    claims = pd.read_csv("claims_data.csv")
    return providers, receivers, food_listings, claims

providers, receivers, food_listings, claims = load_data()

def create_connection():
    conn = sqlite3.connect(":memory:")
    return conn

def load_to_db(conn):
    providers.to_sql("providers", conn, index=False)
    receivers.to_sql("receivers", conn, index=False)
    food_listings.to_sql("food_listings", conn, index=False)
    claims.to_sql("claims", conn, index=False)

conn = create_connection()
load_to_db(conn)

section = st.sidebar.radio("Navigate", ["Providers", "Receivers", "Food Listings", "Claims", "Update Listing", "Add Listing", "Delete Listing", "Queries"])

if section == "Providers":
    st.header("Food Providers")
    st.dataframe(providers)
elif section == "Receivers":
    st.header("Food Receivers")
    st.dataframe(receivers)
elif section == "Food Listings":
    st.header("Food Listings")
    st.dataframe(food_listings)
elif section == "Claims":
    st.header("Food Claims")
    st.dataframe(claims)
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
        new_expiry = st.date_input("Expiry Date")

        if st.button("Update Listing"):
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE food_listings
                    SET Food_Name = ?, Quantity = ?, Expiry_Date = ?
                    WHERE Food_ID = ?
                """, (new_name, new_qty, new_expiry, food_id))
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
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO food_listings (Provider_ID, Food_Name, Quantity, Expiry_Date)
                VALUES (?, ?, ?, ?)
            """, (provider_id, food_name, quantity, expiry_date))
            conn.commit()
            st.success("Listing added successfully.")
        except Exception as e:
            st.error(f"Error: {e}")
elif section == "Delete Listing":
    st.header("Delete Food Listing")

    listings = pd.read_sql_query("SELECT * FROM food_listings", conn)
    st.subheader("Current Food Listings")
    st.dataframe(listings)

    if not listings.empty:
        selected_id = st.selectbox("Select Food to Delete", listings["Food_ID"].apply(lambda x: f"ID: {x} - {listings[listings['Food_ID']==x]['Food_Name'].values[0]}"))
        selected_id_int = int(selected_id.split(":")[1].split("-")[0].strip())
        selected_listing = listings[listings["Food_ID"] == selected_id_int].iloc[0]

        # Show food details
        st.subheader("Food Details")
        st.markdown(f"**Food Name:** {selected_listing['Food_Name']}  \n"
                    f"**Quantity:** {selected_listing['Quantity']}  \n"
                    f"**Expiry Date:** {selected_listing['Expiry_Date']}  \n"
                    f"**Location:** {selected_listing['Location']}  \n"
                    f"**Food Type:** {selected_listing['Food_Type']}  \n"
                    f"**Meal Type:** {selected_listing['Meal_Type']}")

        if st.button("Delete Listing"):
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM food_listings WHERE Food_ID = ?", (selected_id_int,))
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
        "16. Claim Completion Status Breakdown": {
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
