from prediction_using_opencv import AutoCheckoutSystem

# Instantiate the predictor class
client = AutoCheckoutSystem()

# Set configurations
url = "https://autocheckouts.onrender.com/add_item"  # Or your actual deployed URL
client.set_end_point_url(url)
client.set_time_out_threshold(20)
client.set_video_path("video1.mp4")
client.set_user_id("id")

# Run video streaming
client.video_streaming()