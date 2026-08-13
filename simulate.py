import mujoco
import mujoco.viewer
import time
import cv2
import numpy as np
import threading
import re
from llama_cpp import Llama

# ─────────────────────────────────────────────────────────────
# State kontrol robot (tombol yang sedang ditekan)
# ─────────────────────────────────────────────────────────────
ctrl_state = {
    "front": False,
    "back":  False,
    "left":  False,
    "right": False,
    "stand": False,
    "crouch":False,
}
crouching = False

# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────
model    = mujoco.MjModel.from_xml_path("wheeled_dog.xml")
data     = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

def aid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

A = {
    "fl_sh": aid("fl_shoulder_motor"), "fr_sh": aid("fr_shoulder_motor"),
    "bl_sh": aid("bl_shoulder_motor"), "br_sh": aid("br_shoulder_motor"),
    "fl_kn": aid("fl_knee_motor"),     "fr_kn": aid("fr_knee_motor"),
    "bl_kn": aid("bl_knee_motor"),     "br_kn": aid("br_knee_motor"),
    "fl_wh": aid("fl_wheel_motor"),    "fr_wh": aid("fr_wheel_motor"),
    "bl_wh": aid("bl_wheel_motor"),    "br_wh": aid("br_wheel_motor"),
}

# Posisi stabil (tegak) dan jongkok
STAND_SH  =  0.0;  STAND_KN  =  0.0
CROUCH_SH =  0.5;  CROUCH_KN = -0.9
WALK_SPD  =  20.0


def set_legs(sh, kn):
    for k in ["fl_sh","fr_sh","bl_sh","br_sh"]: data.ctrl[A[k]] = sh
    for k in ["fl_kn","fr_kn","bl_kn","br_kn"]: data.ctrl[A[k]] = kn

def set_wheels(vL, vR):
    vL = max(-25.0, min(25.0, vL)); vR = max(-25.0, min(25.0, vR))
    data.ctrl[A["fl_wh"]] = vL; data.ctrl[A["bl_wh"]] = vL
    data.ctrl[A["fr_wh"]] = vR; data.ctrl[A["br_wh"]] = vR

def reset():
    global crouching
    mujoco.mj_resetDataKeyframe(model, data, 0)
    set_legs(STAND_SH, STAND_KN)
    set_wheels(0, 0)
    crouching = False
    for k in ctrl_state: ctrl_state[k] = False
    print("[RESET]")

# ─────────────────────────────────────────────────────────────
# AI Prompting (Qwen Local LLM)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = r"D:\anjing\Qwen3.5-0.8B-Q4_K_M.gguf"
try:
    print("[AI] Memuat model Qwen ke memori (harap tunggu)...")
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    print("[AI] Model berhasil dimuat!")
except Exception as e:
    print(f"[AI] Gagal memuat model: {e}")
    llm = None

SYSTEM_PROMPT = """Kamu adalah AI pengontrol robot anjing 4WD.
User akan memberikan perintah. 
Kamu WAJIB membalas dengan kalimat bahasa Indonesia yang sangat singkat, lalu AKHIRI pesanmu dengan SATU token perintah aksi dalam kurung siku berikut:
[MAJU]
[MUNDUR]
[KIRI]
[KANAN]
[JONGKOK]
[TEGAK]
[STOP]

Contoh respons:
Siap laksanakan, saya maju sekarang. [MAJU]
"""

def execute_ai_command(command):
    global crouching
    print(f"\n>>> [EKSEKUSI ROBOT]: Menerima perintah {command} <<<")
    
    if command == "[STOP]":
        for k in ["front", "back", "left", "right"]: ctrl_state[k] = False
    elif command == "[MAJU]":
        for k in ["front", "back", "left", "right"]: ctrl_state[k] = False
        ctrl_state["front"] = True
    elif command == "[MUNDUR]":
        for k in ["front", "back", "left", "right"]: ctrl_state[k] = False
        ctrl_state["back"] = True
    elif command == "[KIRI]":
        for k in ["front", "back", "left", "right"]: ctrl_state[k] = False
        ctrl_state["left"] = True
    elif command == "[KANAN]":
        for k in ["front", "back", "left", "right"]: ctrl_state[k] = False
        ctrl_state["right"] = True
    elif command == "[JONGKOK]":
        crouching = True
        set_legs(CROUCH_SH, CROUCH_KN)
    elif command == "[TEGAK]":
        crouching = False
        set_legs(STAND_SH, STAND_KN)

def ai_chat_loop():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            user_input = input("\n[Ketik Perintah AI] You: ")
            if user_input.strip() == "": continue
            if user_input.lower() in ["exit", "quit"]:
                print("[AI] Sistem chat dimatikan.")
                break
            
            messages.append({"role": "user", "content": user_input})
            print("AI: ", end="", flush=True)
            
            response = llm.create_chat_completion(
                messages=messages,
                temperature=0.4,
                max_tokens=128,
                stream=True
            )
            
            assistant_text = ""
            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    text = delta["content"]
                    print(text, end="", flush=True)
                    assistant_text += text
            
            print()
            messages.append({"role": "assistant", "content": assistant_text})
            
            # Ekstrak token dengan regex
            found_commands = re.findall(r'\[(MAJU|MUNDUR|KIRI|KANAN|JONGKOK|TEGAK|STOP)\]', assistant_text.upper())
            if found_commands:
                cmd = f"[{found_commands[-1]}]" # ambil token terakhir jika AI nyebut banyak
                execute_ai_command(cmd)
                
        except EOFError:
            break
        except Exception as e:
            print(f"\n[AI] Error saat chatting: {e}")

# Jalankan AI Agent di background thread agar tidak membekukan simulasi
if llm is not None:
    chat_thread = threading.Thread(target=ai_chat_loop, daemon=True)
    chat_thread.start()


# ─────────────────────────────────────────────────────────────
# Layout tombol UI
# ─────────────────────────────────────────────────────────────
BTN_W, BTN_H = 115, 48
PAD  = 8
OX   = 10
OY   = 490

BUTTONS = [
    ("▲  MAJU",    "front",  1, 0),
    ("◄  KIRI",    "left",   0, 1),
    ("▼  MUNDUR",  "back",   1, 1),
    ("KANAN  ►",   "right",  2, 1),
    ("↑  TEGAK",   "stand",  0, 0),
    ("↓  JONGKOK", "crouch", 2, 0),
    ("⟳  RESET",   "reset",  1, 2),
]

def btn_rect(col, row):
    x = OX + col * (BTN_W + PAD)
    y = OY + row * (BTN_H + PAD) + 10
    return (x, y, x + BTN_W, y + BTN_H)

def draw_buttons(img):
    for label, key, col, row in BUTTONS:
        x1, y1, x2, y2 = btn_rect(col, row)
        active = ctrl_state.get(key, False)

        if key == "reset":
            bg = (50, 50, 50); fg = (200, 200, 200)
        elif active:
            bg = (30, 160, 255); fg = (255, 255, 255)
        else:
            bg = (40, 40, 45); fg = (200, 200, 200)

        cv2.rectangle(img, (x1+2, y1+2), (x2+2, y2+2), (10,10,10), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (80,80,90), 1)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx = x1 + (BTN_W - tw) // 2
        ty = y1 + (BTN_H + th) // 2
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, fg, 1, cv2.LINE_AA)

    return img

def draw_status(img):
    moving = any(ctrl_state[k] for k in ["front","back","left","right"])
    label  = "BERGERAK" if moving else ("JONGKOK" if crouching else "DIAM")
    col    = (0,120,255) if moving else (20,140,60)
    w = img.shape[1]
    cv2.rectangle(img, (w-140, 8), (w-8, 42), col, -1)
    cv2.putText(img, label, (w-135, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, "Robot FPV – Mata Robot",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,230,255), 1, cv2.LINE_AA)
    return img

# ─────────────────────────────────────────────────────────────
# Mouse callback
# ─────────────────────────────────────────────────────────────
def on_mouse(event, mx, my, flags, _):
    global crouching
    for label, key, col, row in BUTTONS:
        x1, y1, x2, y2 = btn_rect(col, row)
        if x1 <= mx <= x2 and y1 <= my <= y2:
            if event == cv2.EVENT_LBUTTONDOWN:
                if key == "reset":
                    reset()
                elif key == "stand":
                    crouching = False; set_legs(STAND_SH, STAND_KN)
                elif key == "crouch":
                    crouching = True;  set_legs(CROUCH_SH, CROUCH_KN)
                else:
                    ctrl_state[key] = True
                    if not crouching:
                        set_legs(STAND_SH, STAND_KN)
            elif event == cv2.EVENT_LBUTTONUP:
                if key in ctrl_state:
                    ctrl_state[key] = False
                    # Kembali tegak saat tombol dilepas (jika tidak jongkok)
                    still_moving = any(ctrl_state[k] for k in ["front","back","left","right"] if k != key)
                    if not still_moving and not crouching:
                        set_legs(STAND_SH, STAND_KN)

# ─────────────────────────────────────────────────────────────
# Init & main loop
# ─────────────────────────────────────────────────────────────
reset()
WIN = "Robot FPV – Mata Robot"
cv2.namedWindow(WIN)
cv2.setMouseCallback(WIN, on_mouse)

print("=" * 50)
print("  Wheeled Dog Simulator + LLM AI Control")
print("  - Klik tombol UI di jendela FPV untuk mengontrol manual.")
print("  - Atau ketik perintah teks di terminal ini untuk AI.")
print("=" * 50)

RENDER_DT = 1.0 / 30

with mujoco.viewer.launch_passive(model, data) as viewer:
    last_render = time.time()

    while viewer.is_running():
        # Hitung kecepatan roda dari state tombol
        vL = vR = 0.0
        if ctrl_state["front"]:
            vL += WALK_SPD; vR += WALK_SPD
        if ctrl_state["back"]:
            vL -= WALK_SPD; vR -= WALK_SPD
        if ctrl_state["left"]:
            vL -= WALK_SPD * 0.6; vR += WALK_SPD * 0.6
        if ctrl_state["right"]:
            vL += WALK_SPD * 0.6; vR -= WALK_SPD * 0.6

        set_wheels(vL, vR)

        mujoco.mj_step(model, data)
        viewer.sync()

        now = time.time()
        if now - last_render >= RENDER_DT:
            renderer.update_scene(data, camera="front_cam")
            pixels = renderer.render()
            frame  = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            panel  = np.zeros((210, frame.shape[1], 3), dtype=np.uint8)
            panel[:] = (20, 20, 22)
            canvas = np.vstack([frame, panel])

            canvas = draw_status(canvas)
            canvas = draw_buttons(canvas)
            cv2.imshow(WIN, canvas)
            last_render = now

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

cv2.destroyAllWindows()
