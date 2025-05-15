import streamlit as st
import pandas as pd
import datetime
from database_utils import get_db_connection, load_food_data
from mysql.connector import Error

# ─────────────────────────────────────────────────────────────────────────────
# CRUD OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def add_food():
    """Add a new food listing to the database."""
    st.subheader("➕ Add New Food Listing")

    with st.form("add_food_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            food_name = st.text_input("Food Name")
            quantity = st.number_input("Quantity", min_value=1, value=1)
            expiry = st.date_input("Expiry Date", min_value=datetime.date.today())
            provider_id = st.number_input("Provider ID", min_value=1, value=1)
        
        with col2:
            provider_type = st.selectbox("Provider Type", ["Restaurant", "Grocery Store", "Supermarket", "Bakery", "Hotel", "Farm"])
            location = st.text_input("Location")
            food_type = st.selectbox("Food Type", ["Vegetarian", "Non-Vegetarian", "Vegan", "Dairy", "Gluten-Free", "Organic"])
            meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snacks", "Dessert", "Beverage"])

        submit_button = st.form_submit_button("Add Food", use_container_width=True)

        if submit_button:
            if not food_name or not location:
                st.error("Food name and location are required!")
            else:
                try:
                    conn = get_db_connection()
                    if conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO food_listings (Food_Name, Quantity, Expiry_Date, Provider_ID, Provider_Type, Location, Food_Type, Meal_Type)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                            (food_name, quantity, expiry, provider_id, provider_type, location, food_type, meal_type))
                        conn.commit()
                        conn.close()
                        st.success("✅ Food item added successfully!")

                        # Force refresh data in session state
                        if "food_data" in st.session_state:
                            del st.session_state["food_data"]
                        
                        # Set a flag to trigger rerun using a state change
                        st.session_state.need_rerun = True
                    else:
                        st.error("Failed to connect to database")
                except Error as e:
                    st.error(f"Error adding food item: {str(e)}")

def update_food():
    """Update an existing food listing."""
    st.subheader("✏️ Update Existing Food Listing")
    
    # Load fresh data
    food_data = load_food_data()
    
    # Check if there's data to show
    if food_data.empty:
        st.info("No food listings available to update.")
        return
    
    # Display current data for reference
    with st.expander("Current Food Listings", expanded=True):
        st.dataframe(food_data, use_container_width=True)
    
    try:
        # Make sure Food_ID is properly handled as integer
        food_data['Food_ID'] = food_data['Food_ID'].astype(int)
        
        # Create a list of food options with ID and name
        food_options = [f"ID: {int(row['Food_ID'])} - {row['Food_Name']}" 
                        for _, row in food_data.iterrows()]
        
        if not food_options:
            st.info("No valid food listings available.")
            return
        
        # Select food item to update    
        selected_option = st.selectbox("Select Food to Update", food_options)
        
        # Extract the ID from the selected option
        selected_id = int(selected_option.split("-")[0].replace("ID:", "").strip())
        
        # Find the selected food item
        selected_item = food_data[food_data["Food_ID"] == selected_id]
        
        if selected_item.empty:
            st.error(f"Could not find food with ID {selected_id}")
            return
        
        selected_item = selected_item.iloc[0]
        
        # Display current item details
        st.write("### Current Details:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Food Name:** {selected_item['Food_Name']}")
            st.write(f"**Quantity:** {selected_item['Quantity']}")
        with col2:
            st.write(f"**Expiry Date:** {selected_item['Expiry_Date']}")
            st.write(f"**Location:** {selected_item.get('Location', 'N/A')}")
        with col3:
            st.write(f"**Food Type:** {selected_item.get('Food_Type', 'N/A')}")
            st.write(f"**Meal Type:** {selected_item.get('Meal_Type', 'N/A')}")
        
        st.markdown("---")
        
        # Create update form
        st.write("### Update Details:")
        with st.form(key="update_form"):
            update_col1, update_col2 = st.columns(2)
            
            with update_col1:
                # Handle text fields safely
                current_name = selected_item.get("Food_Name", "")
                updated_name = st.text_input("Food Name", value=current_name)
                
                # Handle quantity safely
                try:
                    current_quantity = int(selected_item["Quantity"])
                except:
                    current_quantity = 1
                updated_quantity = st.number_input("Quantity", min_value=1, value=current_quantity)
                
                # Handle expiry date safely
                try:
                    # Convert string to date if needed
                    if isinstance(selected_item["Expiry_Date"], str):
                        current_expiry = datetime.datetime.strptime(selected_item["Expiry_Date"], "%Y-%m-%d").date()
                    else:
                        current_expiry = selected_item["Expiry_Date"]
                except Exception as e:
                    current_expiry = datetime.date.today()
                
                updated_expiry = st.date_input("Expiry Date", value=current_expiry)
            
            with update_col2:
                # Handle location
                current_location = selected_item.get("Location", "")
                updated_location = st.text_input("Location", value=current_location)
                
                # Food Type
                food_types = ["Vegetarian", "Non-Vegetarian", "Vegan", "Dairy", "Gluten-Free", "Organic"]
                current_food_type = selected_item.get("Food_Type", "Vegetarian")
                try:
                    index = food_types.index(current_food_type)
                except:
                    index = 0
                updated_food_type = st.selectbox("Food Type", food_types, index=index)
                
                # Meal Type
                meal_types = ["Breakfast", "Lunch", "Dinner", "Snacks", "Dessert", "Beverage"]
                current_meal_type = selected_item.get("Meal_Type", "Breakfast")
                try:
                    index = meal_types.index(current_meal_type)
                except:
                    index = 0
                updated_meal_type = st.selectbox("Meal Type", meal_types, index=index)
            
            update_button = st.form_submit_button("Update Food Item", use_container_width=True)
            
            if update_button:
                try:
                    conn = get_db_connection()
                    if conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE food_listings 
                            SET Food_Name = %s, Quantity = %s, Expiry_Date = %s, Location = %s, Food_Type = %s, Meal_Type = %s
                            WHERE Food_ID = %s
                        """, (updated_name, updated_quantity, updated_expiry, updated_location, 
                              updated_food_type, updated_meal_type, selected_id))
                        
                        rows_affected = cursor.rowcount
                        conn.commit()
                        conn.close()
                        
                        if rows_affected > 0:
                            st.success(f"✅ Food item #{selected_id} updated successfully!")
                            # Force refresh data in session state
                            if "food_data" in st.session_state:
                                del st.session_state["food_data"]
                            
                            # Set a flag to trigger rerun using a state change
                            st.session_state.need_rerun = True
                        else:
                            st.warning(f"No changes made to food item #{selected_id}.")
                    else:
                        st.error("Failed to connect to database")
                        
                except Error as e:
                    st.error(f"Error updating food item: {str(e)}")
    
    except Exception as e:
        st.error(f"Error in update function: {str(e)}")

def delete_food():
    """Delete a food listing from the database."""
    st.subheader("🗑️ Delete Food Listing")
    
    # Load fresh data
    food_data = load_food_data()
    
    # Check if there's data to show
    if food_data.empty:
        st.info("No food listings available to delete.")
        return
    
    # Display current data for reference
    with st.expander("Current Food Listings", expanded=True):
        st.dataframe(food_data, use_container_width=True)
    
    try:
        # Make sure Food_ID is properly handled as integer
        food_data['Food_ID'] = food_data['Food_ID'].astype(int)
        
        # Create a list of food options with ID and name
        food_options = [f"ID: {int(row['Food_ID'])} - {row['Food_Name']}" 
                        for _, row in food_data.iterrows()]
        
        if not food_options:
            st.info("No valid food listings available.")
            return
        
        # Select food item to delete  
        selected_option = st.selectbox("Select Food to Delete", food_options)
        
        # Extract the ID from the selected option
        selected_id = int(selected_option.split("-")[0].replace("ID:", "").strip())
        
        # Find the selected food item
        selected_item = food_data[food_data["Food_ID"] == selected_id]
        
        if selected_item.empty:
            st.error(f"Could not find food with ID {selected_id}")
            return
        
        selected_item = selected_item.iloc[0]
        
        # Display food details in columns
        st.write("### Food Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Food Name:** {selected_item['Food_Name']}")
            st.write(f"**Quantity:** {selected_item['Quantity']}")
        
        with col2:
            st.write(f"**Expiry Date:** {selected_item['Expiry_Date']}")
            st.write(f"**Location:** {selected_item.get('Location', 'N/A')}")
        
        with col3:
            st.write(f"**Food Type:** {selected_item.get('Food_Type', 'N/A')}")
            st.write(f"**Meal Type:** {selected_item.get('Meal_Type', 'N/A')}")
        
        st.markdown("---")
        
        # Confirm deletion with a warning
        st.warning("⚠️ Are you sure you want to delete this food item? This action cannot be undone.")
        
        delete = st.button("Yes, Delete This Food Item", type="primary", use_container_width=True)
        
        if delete:
            try:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM food_listings WHERE Food_ID = %s", (selected_id,))
                    
                    rows_affected = cursor.rowcount
                    conn.commit()
                    conn.close()
                    
                    if rows_affected > 0:
                        st.success(f"✅ Food item #{selected_id} deleted successfully!")
                        # Force refresh data in session state
                        if "food_data" in st.session_state:
                            del st.session_state["food_data"]
                        
                        # Set a flag to trigger rerun using a state change
                        st.session_state.need_rerun = True
                    else:
                        st.warning(f"No food item with ID #{selected_id} was found or deleted.")
                else:
                    st.error("Failed to connect to database")
                    
            except Error as e:
                st.error(f"Error deleting food item: {str(e)}")
    
    except Exception as e:
        st.error(f"Error in delete function: {str(e)}")
