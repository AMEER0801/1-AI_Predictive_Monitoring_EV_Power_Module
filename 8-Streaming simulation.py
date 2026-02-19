# Streaming simulation producing frames and saving GIF
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np, pandas as pd
import joblib
from IPython.display import HTML, Image, display
import imageio
import os

# Load model (use retrained or your .pkl)
model = joblib.load('rf_power_module_model.pkl')

# Create a mixed stream - shuffle some normal + injected abnormal events
stream_df = X_test.copy().reset_index(drop=True)
# To make visible anomalies, insert some abnormal rows at random places
abn_examples = data[data['label']==1].sample(30, random_state=1).drop('label',axis=1).reset_index(drop=True)
# Insert abnormalities at 10 random positions
positions = np.linspace(10, 90, len(abn_examples)).astype(int)
for i, pos in enumerate(positions):
    stream_df = pd.concat([stream_df.iloc[:pos+i], abn_examples.iloc[[i]], stream_df.iloc[pos+i:]]).reset_index(drop=True)

# Prepare figure
fig, ax = plt.subplots(figsize=(8,3))
ax.set_xlim(0, 100)
ax.set_ylim(-0.05, 1.05)
ax.set_title("Real-Time Risk Score Simulation")
ax.set_xlabel("Time Step")
ax.set_ylabel("Failure Risk Probability")
line, = ax.plot([], [], lw=2)
xdata, ydata = [], []

# Animation function
def animate(i):
    row = stream_df.iloc[i]
    # Ensure the row is passed as a DataFrame with column names to avoid warnings
    prob = model.predict_proba(pd.DataFrame([row], columns=X.columns))[0,1]
    xdata.append(i)
    ydata.append(prob)
    line.set_data(xdata, ydata)
    return line,

# Create animation for first 100 steps
anim = animation.FuncAnimation(fig, animate, frames=100, interval=80, blit=True)

# Save as MP4 (requires ffmpeg) OR fallback to GIF using imageio
mp4_path = "ev_power_risk_stream.mp4"
gif_path = "ev_power_risk_stream.gif"
try:
    anim.save(mp4_path, writer='ffmpeg', fps=12)
    print("Saved mp4:", mp4_path)
    if os.path.exists(mp4_path):
        file_size_bytes = os.path.getsize(mp4_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"MP4 file size: {file_size_mb:.2f} MB")
        if file_size_bytes < 1000: # Arbitrary small threshold for potentially empty video
            print("Warning: The generated MP4 file is very small, it might be empty or corrupted.")
    display(HTML(f"<video controls src='{mp4_path}' width=700></video>"))
except Exception as e:
    print("ffmpeg not available, saving GIF fallback:", e)
    # Fallback: render frames and write gif
    frames = []
    for i in range(100):
        animate(i)
        # draw canvas and convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
    imageio.mimsave(gif_path, frames, fps=12)
    print("Saved gif:", gif_path)
    display(Image(gif_path))
