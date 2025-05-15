#database_utils.py
import streamlit as st
import pandas as pd
import datetime
import mysql.connector
from mysql.connector import Error

def get_db_connection():
    """Create a database connection to TiDB Cloud."""
    try:
        # TiDB Cloud connection parameters
        db_config = {
            'host': 'gateway01.ap-southeast-1.prod.aws.tidbcloud.com',
            'port': 4000,
            'user': '4ZrUUWFVXrLrXUg.root',
            'password': 'wIdAtRb3s0xhjPhL',
            'database': 'test'
        }
        
        # Connect to TiDB Cloud
        conn = mysql.connector.connect(**db_config)
        
        # Enable autocommit
        conn.autocommit = True
        
        return conn
    except Error as e:
        st.error(f"TiDB connection error: {str(e)}")
        return None

def init_database():
    """Initialize the database with tables and sample data."""
    try:
        conn = get_db_connection()
        if not conn:
            st.error("Failed to establish database connection")
            return False
            
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS food_listings (
            Food_ID INT AUTO_INCREMENT PRIMARY KEY,
            Food_Name VARCHAR(255) NOT NULL,
            Quantity INT NOT NULL,
            Expiry_Date DATE NOT NULL,
            Provider_ID INT NOT NULL,
            Provider_Type VARCHAR(100),
            Location VARCHAR(255),
            Food_Type VARCHAR(100),
            Meal_Type VARCHAR(100),
            Listed_Date DATE DEFAULT (CURRENT_DATE)
        )
        ''')
        
        # Check if the table is empty, if so, add sample data
        cursor.execute("SELECT COUNT(*) FROM food_listings")
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Add sample data
            today = datetime.date.today()
            
            # Sample data with varied expiry dates
            sample_data = [
                ("Fresh Vegetables", 10, (today + datetime.timedelta(days=3)).isoformat(), 1, "Farm", "Downtown Market", "Vegetarian", "Dinner"),
                ("Bread Loaves", 20, (today + datetime.timedelta(days=1)).isoformat(), 2, "Bakery", "Main Street", "Vegan", "Breakfast"),
                ("Milk Cartons", 15, (today + datetime.timedelta(days=5)).isoformat(), 3, "Grocery Store", "North District", "Dairy", "Breakfast"),
                ("Cooked Pasta", 25, (today + datetime.timedelta(days=2)).isoformat(), 4, "Restaurant", "Downtown", "Vegetarian", "Lunch"),
                ("Fresh Fruits", 30, (today + datetime.timedelta(days=4)).isoformat(), 1, "Farm", "East Market", "Organic", "Snacks"),
                ("Chicken Curry", 8, (today + datetime.timedelta(days=1)).isoformat(), 5, "Restaurant", "South District", "Non-Vegetarian", "Dinner"),
                ("Pastries", 12, (today + datetime.timedelta(days=2)).isoformat(), 2, "Bakery", "West End", "Vegetarian", "Dessert"),
                ("Rice Bags", 5, (today + datetime.timedelta(days=30)).isoformat(), 6, "Supermarket", "Central Area", "Gluten-Free", "Dinner"),
                ("Fresh Juice", 18, (today + datetime.timedelta(days=3)).isoformat(), 7, "Juice Bar", "Market Square", "Organic", "Beverage"),
                ("Vegetable Soup", 15, (today + datetime.timedelta(days=2)).isoformat(), 4, "Restaurant", "Downtown", "Vegan", "Lunch")
            ]
            
            # MySQL uses %s as placeholder instead of ? in SQLite
            cursor.executemany('''
            INSERT INTO food_listings 
            (Food_Name, Quantity, Expiry_Date, Provider_ID, Provider_Type, Location, Food_Type, Meal_Type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', sample_data)
            
            conn.commit()
            st.sidebar.success("Database initialized with sample data!")
        
        conn.close()
        return True
    except Error as e:
        st.error(f"Database initialization error: {str(e)}")
        return False

def load_food_data():
    """Load food listings from the database."""
    try:
        conn = get_db_connection()
        if not conn:
            return pd.DataFrame()
        
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM food_listings ORDER BY Food_ID DESC"
        cursor.execute(query)
        
        # Fetch all rows as dictionaries
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to DataFrame
        data = pd.DataFrame(rows) if rows else pd.DataFrame()
        
        # Convert date strings to datetime objects
        if not data.empty:
            if 'Expiry_Date' in data.columns:
                data['Expiry_Date'] = pd.to_datetime(data['Expiry_Date']).dt.date
            
            if 'Listed_Date' in data.columns:
                data['Listed_Date'] = pd.to_datetime(data['Listed_Date']).dt.date
                
        return data
    except Error as e:
        st.error(f"Error loading food data: {str(e)}")
        return pd.DataFrame()

def run_query(query):
    """Execute a custom SQL query and return results as a DataFrame."""
    try:
        conn = get_db_connection()
        if not conn:
            return pd.DataFrame()
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        
        # Fetch all rows as dictionaries
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to DataFrame
        data = pd.DataFrame(rows) if rows else pd.DataFrame()
        return data
    except Error as e:
        st.error(f"Error running query: {str(e)}")
        return pd.DataFrame()
