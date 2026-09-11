# keeping this as reference for how to screenshot the bottom third
import mss.tools

with mss.MSS() as sct:
    monitor = sct.monitors[1]

    screen_width = monitor["width"]
    screen_height = monitor["height"]


    mregion = {"top": int(screen_height)*1/3, "left": 0, "width": screen_width, "height": (int(screen_height)*1/3)*2}

    funny = sct.grab(mregion)
    mss.tools.to_png(funny.rgb, funny.size, output="screenshot.png")


print(screen_width, screen_height)

