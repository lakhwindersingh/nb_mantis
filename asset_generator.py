import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ================================
# CONFIG
# ================================
OUTPUT_DIR = "assets"
RESOLUTIONS = [32, 64, 128]
COLORS = {
    "red": (255, 0, 0, 255),
    "blue": (0, 0, 255, 255),
    "green": (0, 255, 0, 255),
    "black": (0, 0, 0, 255),
    "white": (255, 255, 255, 255),
    "yellow": (255, 255, 0),  # Add this
    "brown": (139, 69, 19),  # Add this
    "gray": (128, 128, 128)  # Add this

}

# Utility: ensure folder exists
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Utility: save both color + grayscale versions
def save_with_grayscale(img, base_path):
    img.save(base_path + ".png")
    gray = img.convert("L")
    gray.save(base_path + "_gray.png")

# ================================
# CHARACTER SPRITES (very simple shapes for testing)
# ================================
def generate_character_sprites():
    base = os.path.join(OUTPUT_DIR, "characters")
    ensure_dir(base)
    for style in ["humanoid_realistic", "humanoid_cartoon"]:
        style_dir = os.path.join(base, style)
        for action in ["idle", "walk", "jump"]:
            action_dir = os.path.join(style_dir, action)
            ensure_dir(action_dir)
            for res in RESOLUTIONS:
                # Simple stick figure placeholder
                for frame in range(1, 5 if action == "walk" else 3 if action == "jump" else 1):
                    img = Image.new("RGBA", (res, res), (0,0,0,0))
                    d = ImageDraw.Draw(img)
                    # Body
                    d.line([(res//2, res//4), (res//2, 3*res//4)], fill=COLORS["black"], width=max(1,res//16))
                    # Head
                    d.ellipse([res//3, 0, 2*res//3, res//3], fill=COLORS["blue"])
                    # Arms / Legs (simple variation per frame)
                    offset = (frame % 2) * (res//8)
                    d.line([(res//2, res//2), (res//2 - offset, res//2 + res//4)], fill=COLORS["red"], width=2)
                    d.line([(res//2, res//2), (res//2 + offset, res//2 + res//4)], fill=COLORS["red"], width=2)

                    save_with_grayscale(img, os.path.join(action_dir, f"{res}px_frame{frame}"))

# ================================
# OBJECT PROPS
# ================================
def generate_objects():
    objects = ["treasure_chest", "key", "potion", "sword", "bow", "door", "lever", "button", "tree", "rock", "building", "bridge"]
    base = os.path.join(OUTPUT_DIR, "objects")
    for obj in objects:
        obj_dir = os.path.join(base, obj)
        ensure_dir(obj_dir)
        for res in RESOLUTIONS:
            img = Image.new("RGBA", (res, res), (0,0,0,0))
            d = ImageDraw.Draw(img)
            if obj == "treasure_chest":
                d.rectangle([res//8, res//3, 7*res//8, 7*res//8], fill=COLORS["brown"] if "brown" in COLORS else COLORS["red"])
                d.rectangle([res//8, res//3, 7*res//8, res//2], fill=COLORS["yellow"])
            elif obj == "key":
                d.rectangle([res//3, res//2, 2*res//3, res//2 + res//4], fill=COLORS["yellow"])
            elif obj == "potion":
                d.ellipse([res//4, res//4, 3*res//4, 3*res//4], fill=COLORS["blue"])
            elif obj == "sword":
                d.line([(res//2, res//4), (res//2, 3*res//4)], fill=COLORS["gray"] if "gray" in COLORS else COLORS["white"], width=3)
            else:
                d.rectangle([res//4, res//4, 3*res//4, 3*res//4], outline=COLORS["black"], width=2)
            save_with_grayscale(img, os.path.join(obj_dir, f"{res}px"))

# ================================
# UI ELEMENTS
# ================================
def generate_ui():
    base = os.path.join(OUTPUT_DIR, "ui")
    for ui_cat in ["health_bars", "buttons", "icons"]:
        ui_dir = os.path.join(base, ui_cat)
        ensure_dir(ui_dir)
        for res in RESOLUTIONS:
            img = Image.new("RGBA", (res*2, res//2), (0,0,0,0))
            d = ImageDraw.Draw(img)
            if ui_cat == "health_bars":
                d.rectangle([0,0,res*2,res//2], outline=COLORS["black"], width=2)
                d.rectangle([0,0,res,res//2], fill=COLORS["red"])
            elif ui_cat == "buttons":
                d.rectangle([0,0,res,res//2], fill=COLORS["blue"], outline=COLORS["black"], width=2)
            elif ui_cat == "icons":
                d.ellipse([0,0,res//2,res//2], fill=COLORS["green"])
            save_with_grayscale(img, os.path.join(ui_dir, f"{res}px"))

# ================================
# TEST PATTERNS
# ================================
def generate_test_patterns():
    base = os.path.join(OUTPUT_DIR, "test_patterns")
    ensure_dir(base)

    # Checkerboards
    cb_dir = os.path.join(base, "checkerboards")
    ensure_dir(cb_dir)
    for res in RESOLUTIONS:
        img = Image.new("RGBA", (res,res), (0,0,0,0))
        d = ImageDraw.Draw(img)
        step = res//8
        for y in range(0, res, step):
            for x in range(0, res, step):
                color = COLORS["black"] if (x//step + y//step) % 2 == 0 else COLORS["white"]
                d.rectangle([x,y,x+step,y+step], fill=color)
        save_with_grayscale(img, os.path.join(cb_dir, f"checker_{res}px"))

    # Color charts (RGB gradient)
    cc_dir = os.path.join(base, "color_charts")
    ensure_dir(cc_dir)
    for res in RESOLUTIONS:
        arr = np.zeros((res,res,3), dtype=np.uint8)
        for y in range(res):
            for x in range(res):
                arr[y,x] = [x*255//res, y*255//res, 128]
        img = Image.fromarray(np.dstack([arr, 255*np.ones((res,res), dtype=np.uint8)]), 'RGBA')
        save_with_grayscale(img, os.path.join(cc_dir, f"rgbchart_{res}px"))

# ================================
# RUN ALL
# ================================
if __name__ == "__main__":
    generate_character_sprites()
    generate_objects()
    generate_ui()
    generate_test_patterns()
    print("✅ Asset pack generated in 'assets/' folder")