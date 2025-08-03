import streamlit as st
import os
import getpass

st.set_page_config(page_title="System Diagnostic", layout="wide")
st.header("🔬 Cloud Environment Diagnostic")

st.info("This is a diagnostic tool to check the permissions and environment of the Hugging Face Space.")

try:
    # --- Check 1: Environment Variables ---
    st.subheader("1. Environment Variables")
    streamlit_home_env = os.getenv('STREAMLIT_HOME')
    st.write(f"**`STREAMLIT_HOME` variable is set to:** `{streamlit_home_env}`")
    if not streamlit_home_env:
        st.warning("Warning: `STREAMLIT_HOME` is not set.")

    # --- Check 2: Current Working Directory ---
    st.subheader("2. Current Directory and User")
    cwd = os.getcwd()
    st.write(f"**Current Working Directory:** `{cwd}`")
    st.write(f"**Running as user:** `{getpass.getuser()}`")

    # --- Check 3: Directory Permissions Test ---
    st.subheader("3. Writable Directory Test")
    
    # Path to test
    test_dir = os.path.join(cwd, ".streamlit")
    st.write(f"**Attempting to create directory at:** `{test_dir}`")
    
    try:
        os.makedirs(test_dir, exist_ok=True)
        st.success(f"✅ Successfully created or found the directory: `{test_dir}`")
        
        # Test writing a file
        test_file_path = os.path.join(test_dir, "test.txt")
        with open(test_file_path, "w") as f:
            f.write("This is a test.")
        st.success(f"✅ Successfully wrote a test file to: `{test_file_path}`")

    except Exception as e:
        st.error(f"❌ FAILED to create or write to the directory.")
        st.exception(e)

except Exception as e:
    st.error("An unexpected error occurred during diagnostics.")
    st.exception(e)