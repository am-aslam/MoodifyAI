# Deploiment.py
import streamlit as st
import torch
import numpy as np
import pygame
from flappy.game import FlappyBirdEnv
from flappy.dqn import QNetwork
from flappy import config
from streamlit_autorefresh import st_autorefresh   # pip install streamlit-autorefresh

# -------------------- Page Setup -------------------- #
st.set_page_config(page_title="Flappy Bird AI", layout="centered", page_icon="🎮")

st.markdown(
    """
    <style>
    body, .stApp {
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: #000;
    }
    .game-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        background-color: #111;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
        margin-top: 20px;
    }
    .score {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 10px;
        color: white;
    }
    .controls {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin: 15px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'>🎮 Flappy Bird — Manual or AI Mode 🤖</h1>", unsafe_allow_html=True)

# -------------------- Init Session State -------------------- #
if "env" not in st.session_state:
    st.session_state.env = FlappyBirdEnv(render=False)  # headless pygame surface
    st.session_state.state = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.score = 0

env = st.session_state.env
state = st.session_state.state

# -------------------- Mode Selection -------------------- #
st.markdown("<h3 style='text-align: center;'>Select Game Mode</h3>", unsafe_allow_html=True)
mode = st.radio("", ["Manual Play", "AI Play"], horizontal=True, index=0)

# -------------------- Load Model for AI -------------------- #
device = 'cuda' if config.DEVICE == 'cuda' and torch.cuda.is_available() else 'cpu'
net = QNetwork(5, 2).to(device)
try:
    ckpt = torch.load("models/best.pt", map_location=device)
    net.load_state_dict(ckpt['model'])
    ai_ready = True
except:
    ai_ready = False
    st.warning("⚠️ No trained model found. Train first using `python -m flappy.train`.")

# -------------------- Controls -------------------- #
st.markdown('<div class="controls">', unsafe_allow_html=True)
if st.button("🔄 Reset Game", key="reset_game"):
    st.session_state.state = env.reset()
    st.session_state.done = False
    st.session_state.score = 0
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Manual control
manual_flap = False
if mode == "Manual Play":
    st.markdown('<div class="controls">', unsafe_allow_html=True)
    if st.button("⬆️ Flap!", key="manual_flap"):
        manual_flap = True
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Game Step -------------------- #
action = 0
if mode == "Manual Play" and manual_flap:
    action = 1
elif mode == "AI Play" and ai_ready:
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = net(s)
        action = int(torch.argmax(q, dim=1).item())

if not st.session_state.done:
    new_state, reward, done, info = env.step(action)
    st.session_state.state = new_state
    st.session_state.done = done
    st.session_state.score = info.get("score", 0)

# -------------------- Render Game -------------------- #
st.markdown('<div class="game-container">', unsafe_allow_html=True)

frame = pygame.surfarray.array3d(env.screen)  # Always valid (offscreen surface)
frame = frame.swapaxes(0, 1)  # (W,H,C) → (H,W,C)

# Resize frame for Streamlit
target_height = 450
aspect_ratio = frame.shape[1] / frame.shape[0]
target_width = int(target_height * aspect_ratio)

st.image(frame, width=target_width, use_container_width=False)
st.markdown(f'<div class="score">Score: {st.session_state.score}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Auto Refresh -------------------- #
st_autorefresh(interval=50, key="flappy_refresh")  # ~20 FPS
