#app.py
import streamlit as st
import pandas as pd
import datetime
from database_utils import get_db_connection, init_database, load_food_data
from crud_operations import add_food, update_food, delete_food
from query_operations import show_queries

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # App Title and Header
    st.set_page_config(
        page_title="Food Wastage Management System",
        page_icon="🍱",
        layout="wide"
    )
    
    st.title("🍱 Local Food Wastage Management System")
    st.markdown("""
        Welcome to the platform that connects food providers with receivers to reduce food waste and help those in need.
        Use the sidebar to navigate between different operations.
    """)
    
    # Initialize the database
    if "db_initialized" not in st.session_state:
        success = init_database()
        st.session_state.db_initialized = success
        if success:
            st.sidebar.success("✅ Database initialized!")
        else:
            st.sidebar.error("❌ Database initialization failed!")

    # Initialize rerun flag if not present
    if "need_rerun" not in st.session_state:
        st.session_state.need_rerun = False
        
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    
    # Add sidebar info
    st.sidebar.info("This system helps manage food donations and reduce waste by connecting providers with receivers.")
    
    # Navigation options
    st.sidebar.header("🔧 Select Operation")
    operation = st.sidebar.radio("Go to", ["Add Food", "Update Food", "Delete Food", "Run SQL Queries"])
    
    # Clear session state when switching operations
    if "last_operation" not in st.session_state:
        st.session_state.last_operation = None
    
    # Check if operation changed and clear cached data
    if st.session_state.last_operation != operation:
        if "food_data" in st.session_state:
            del st.session_state["food_data"]
        st.session_state.last_operation = operation
    
    try:
        # Create a container for the main content
        with st.container():
            if operation == "Add Food":
                add_food()
            elif operation == "Update Food":
                update_food()
            elif operation == "Delete Food":
                delete_food()
            else:
                show_queries()
    except Exception as e:
        st.error(f"Error performing operation: {str(e)}")

    # Show current food listings (for reference)
    with st.expander("Show all current food listings", expanded=False):
        try:
            current_data = load_food_data()
            if current_data.empty:
                st.info("No food listings available.")
            else:
                st.dataframe(current_data, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading food data: {str(e)}")
    
    # Database status in sidebar
    db_test = get_db_connection()
    if db_test:
        st.sidebar.success("Database connection: Success ✓")
        db_test.close()
    else:
        st.sidebar.error("Database connection: Failed ✗")
    
    # Check if we need to rerun the app due to a state change
    if st.session_state.need_rerun:
        st.session_state.need_rerun = False
        st.rerun()

# Run the main app
if __name__ == "__main__":
    main()
