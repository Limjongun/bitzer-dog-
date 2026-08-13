import mujoco
import numpy as np
import cv2
import time

def main():
    # Load model and data
    model = mujoco.MjModel.from_xml_path("wheeled_dog.xml")
    data = mujoco.MjData(model)
    
    # Inisialisasi offscreen renderer untuk mengambil gambar dari kamera
    # Ukuran resolusi bisa diubah sesuai kebutuhan (contoh: 640x480)
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Menyiapkan indeks motor untuk dikontrol
    wheel_indices = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fl_wheel_motor"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fr_wheel_motor"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "bl_wheel_motor"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "br_wheel_motor")
    ]
    shoulder_indices = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fl_shoulder_motor"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fr_shoulder_motor"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "bl_shoulder_motor"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "br_shoulder_motor")
    ]
    knee_indices = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fl_knee_motor"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fr_knee_motor"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "bl_knee_motor"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "br_knee_motor")
    ]
    
    # Setel postur dasar robot
    for idx in shoulder_indices:
        data.ctrl[idx] = 0.5
    for idx in knee_indices:
        data.ctrl[idx] = -1.0
    for idx in wheel_indices:
        data.ctrl[idx] = 10.0 # Roda bergerak maju
        
    print("Mulai simulasi dan capture kamera...")
    print("Tekan 'q' pada jendela OpenCV untuk keluar.")
    
    # Target 30 FPS untuk preview kamera
    fps = 30
    frame_time = 1.0 / fps
    
    # Timestep adalah 0.002s, jadi butuh banyak step per frame kamera
    steps_per_frame = int(frame_time / model.opt.timestep)

    while True:
        # 1. Jalankan fisika (step simulasi)
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
            
        # 2. Update renderer scene dari kamera yang dipasang di robot
        renderer.update_scene(data, camera="front_cam")
        
        # 3. Ekstrak RGB array (Numpy array)
        # Array inilah yang bisa Anda "import" atau gunakan untuk Machine Learning/Computer Vision!
        pixels = renderer.render()
        
        # 4. Tampilkan preview (OpenCV menggunakan format BGR, jadi dikonversi dulu)
        bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.imshow("Robot FPV Camera (Real-time Preview)", bgr_pixels)
        
        # Tekan 'q' untuk berhenti
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
