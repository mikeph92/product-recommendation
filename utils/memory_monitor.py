import streamlit as st
import psutil
import os
import gc
import numpy as np
import pandas as pd

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024  # Convert to MB
    return memory_usage_mb

def display_memory_usage():
    """Display current memory usage in the Streamlit app"""
    memory_usage = get_memory_usage()
    
    # Create a progress bar for memory usage
    memory_limit = 2048  # Assuming 2GB limit, adjust as needed
    memory_percent = min(100, (memory_usage / memory_limit) * 100)
    
    # Color coding based on memory usage
    if memory_percent < 50:
        color = "green"
    elif memory_percent < 80:
        color = "orange"
    else:
        color = "red"
    
    st.sidebar.markdown(f"""
    <div style="margin-top: 20px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
        <h4>Memory Usage</h4>
        <div style="background-color: #e0e0e0; height: 20px; border-radius: 10px; overflow: hidden;">
            <div style="background-color: {color}; width: {memory_percent}%; height: 100%;"></div>
        </div>
        <p>{memory_usage:.1f} MB / {memory_limit} MB ({memory_percent:.1f}%)</p>
    </div>
    """, unsafe_allow_html=True)

def clear_memory():
    """Clear memory by running garbage collection"""
    gc.collect()
    
    # Clear Streamlit cache if needed
    st.cache_data.clear()
    
    # Get memory after cleanup
    memory_after = get_memory_usage()
    
    return memory_after

def add_memory_cleanup_button():
    """Add a button to manually clear memory"""
    if st.sidebar.button("🧹 Clear Memory Cache"):
        memory_before = get_memory_usage()
        memory_after = clear_memory()
        memory_saved = memory_before - memory_after
        
        st.sidebar.success(f"Memory cleared! Saved {memory_saved:.1f} MB")
        st.experimental_rerun() 