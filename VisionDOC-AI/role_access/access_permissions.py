import box
import yaml
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def load_users():
    with open('role_access/users.yml', 'r', encoding='utf8') as ymlfile:
        users_cfg = box.Box(yaml.safe_load(ymlfile))
    return users_cfg.users

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate(username, password):
    users = load_users()
    user = users.get(username)
    if user and verify_password(password, user.password):
        return user.role
    return None



def access():
    print("=====AUTHENTICATE=====")
    username = input('user: ')
    password = input('password: ')

    role = authenticate(username, password)

    if not role:
        print('failed!')
        exit(1)
    return role

def access_streamlit():
    import streamlit as st

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.role = None

    if not st.session_state.authenticated:
        st.subheader("🔐 Please login to continue")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            role = authenticate(username, password)
            if role:
                st.session_state.authenticated = True
                st.session_state.role = role
            else:
                st.error("❌ Authentication failed. Please try again.")

    return st.session_state.authenticated, st.session_state.role