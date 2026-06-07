# -*- coding: utf-8 -*-
from pathlib import Path
import json
import re
import time
import sys

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as e:
    print("\nTHIEU THU VIEN:", e)
    print("Hay chay file tai thu vien , sau đó restart kernel Spyder.")
    raise


# ========================= 
# 1. CAU HINH PROJECT
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_faces"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

IMG_SIZE = 200
CAMERA_IDS = [0, 1, 2]
DEFAULT_IMAGES_PER_PERSON = 200

UNKNOWN_THRESHOLD = 50.0


# =========================
# 2. HAM DUNG CHUNG
# =========================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9_\- ]+", "", name)
    name = name.replace(" ", "_")
    return name or "person"


def init_folders():
    ensure_dir(DATASET_DIR)
    ensure_dir(MODEL_DIR)
    ensure_dir(OUTPUT_DIR)


def open_camera(camera_ids=CAMERA_IDS):
    for cam_id in camera_ids:
        for backend in [cv2.CAP_DSHOW, 0]:
            if backend != 0:
                cap = cv2.VideoCapture(cam_id, backend)
            else:
                cap = cv2.VideoCapture(cam_id)

            if cap.isOpened():
                print(f"Da mo camera ID = {cam_id}")
                return cap
            cap.release()

    raise RuntimeError(
        "Khong mo duoc camera. Hay kiem tra webcam, quyen camera, "
        "hoac doi CAMERA_IDS trong dau file."
    )


def get_face_detector():
    """Dung Haar Cascade co san trong OpenCV de phat hien khuon mat."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError("Khong tai duoc Haar Cascade. Hay cai lai opencv-contrib-python.")
    return detector


def detect_faces(gray_img, detector):
    faces = detector.detectMultiScale(
        gray_img,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(70, 70),
    )
    faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
    return faces


def preprocess_face(gray_img, box, img_size=IMG_SIZE):
    x, y, w, h = box
    face = gray_img[y:y+h, x:x+w]
    face = cv2.resize(face, (img_size, img_size))
    face = cv2.equalizeHist(face)
    return face




def hien_thi_anh_vua_man_hinh(window_name, image, max_width=900, max_height=520):
    h, w = image.shape[:2]

    scale = min(max_width / w, max_height / h, 1.0)

    new_w = int(w * scale)
    new_h = int(h * scale)

    image_show = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, new_w, new_h)
    cv2.imshow(window_name, image_show)


def check_cv2_face():
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "Ban dang thieu module cv2.face.\n"
            "Cach sua: chay file 00_install_packages.py, sau do restart kernel Spyder.\n"
            "Thu vien can dung la: opencv-contrib-python"
        )


# =========================
# 3. THU THAP DU LIEU
# =========================
def collect_faces():
    init_folders()

    person_name = input("Nhap ten nguoi can thu thap du lieu: ")
    person_name = safe_name(person_name)

    target_text = input(
        f"So anh muon chup cho {person_name} "
        f"(Enter = {DEFAULT_IMAGES_PER_PERSON}): "
    ).strip()
    target_count = int(target_text) if target_text else DEFAULT_IMAGES_PER_PERSON

    save_dir = DATASET_DIR / person_name
    ensure_dir(save_dir)

    existing = len(list(save_dir.glob("*.jpg")))
    count = existing

    detector = get_face_detector()
    cap = open_camera()

    print("\nHUONG DAN CHUP DU LIEU:")
    print("- Bam S de bat dau / tam dung chup anh")
    print("- Bam Q de thoat")
    print("- Nen xoay nhe mat trai/phai, len/xuong, doi bieu cam va anh sang")
    print("- Nen chup 150-300 anh moi nguoi")

    saving = False
    last_save_time = 0.0
    save_interval = 0.12

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Khong doc duoc khung hinh tu camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)

        if faces:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if saving and count < target_count:
                now = time.time()
                if now - last_save_time >= save_interval:
                    face_img = preprocess_face(gray, faces[0])
                    filename = save_dir / f"{person_name}_{count+1:04d}.jpg"
                    cv2.imwrite(str(filename), face_img)
                    count += 1
                    last_save_time = now

        status = "DANG CHUP" if saving else "TAM DUNG"
        cv2.putText(frame, f"Nguoi: {person_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Trang thai: {status}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Anh: {count}/{target_count}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "S: start/pause | Q: quit", (10, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Thu thap du lieu khuon mat", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord('s'), ord('S')]:
            saving = not saving
        elif key in [ord('q'), ord('Q')]:
            break

        if count >= target_count:
            print(f"Da chup du {target_count} anh cho {person_name}.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDa luu du lieu tai:", save_dir)


# =========================
# 4. DOC DATASET
# =========================
def load_dataset():
    """Doc dataset trong thu muc dataset_faces/person_name/*.jpg."""
    images = []
    labels = []
    label_map = {}

    if not DATASET_DIR.exists():
        raise RuntimeError("Chua co thu muc dataset_faces. Hay thu thap du lieu truoc.")

    person_dirs = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])
    if len(person_dirs) < 2:
        raise RuntimeError(
            "Can toi thieu 2 nguoi de train nhan dien khuon mat. "
            "Hay chup them du lieu cho nguoi khac."
        )

    for label_id, person_dir in enumerate(person_dirs):
        label_map[label_id] = person_dir.name
        image_files = sorted(list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")))

        if len(image_files) < 20:
            print(f"Canh bao: {person_dir.name} chi co {len(image_files)} anh. Nen co >= 100 anh/nguoi.")

        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.equalizeHist(img)
            images.append(img)
            labels.append(label_id)

    if len(images) == 0:
        raise RuntimeError("Khong doc duoc anh nao trong dataset_faces.")

    return images, np.array(labels, dtype=np.int32), label_map


# =========================
# 5. HUAN LUYEN & DANH GIA
# =========================
def make_train_test_split(labels, test_ratio=0.2, seed=42):
    """
    Tu chia train/test theo tung nguoi, khong dung scikit-learn.
    Moi lop se co it nhat 1 anh test neu so anh cua lop do >= 2.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_idx = []
    test_idx = []

    for label_id in sorted(set(labels.tolist())):
        idx = np.where(labels == label_id)[0]
        rng.shuffle(idx)

        if len(idx) < 2:
            train_idx.extend(idx.tolist())
            continue

        n_test = max(1, int(round(len(idx) * test_ratio)))
        n_test = min(n_test, len(idx) - 1)

        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx, dtype=np.int32), np.array(test_idx, dtype=np.int32)


def build_confusion_matrix(y_true, y_pred, labels_sorted):
    """Tao confusion matrix bang numpy, khong dung sklearn."""
    label_to_pos = {label: i for i, label in enumerate(labels_sorted)}
    cm = np.zeros((len(labels_sorted), len(labels_sorted)), dtype=np.int32)
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_pos and pred_label in label_to_pos:
            cm[label_to_pos[true_label], label_to_pos[pred_label]] += 1
    return cm


def make_report_text(y_true, y_pred, labels_sorted, label_map, cm):
    lines = []
    lines.append("Ten nguoi                 Precision   Recall   F1-score   Support")
    lines.append("-" * 68)

    total_correct = 0
    total_samples = len(y_true)

    for i, label_id in enumerate(labels_sorted):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        total_correct += tp

        name = label_map[label_id][:22]
        lines.append(f"{name:<24} {precision:>8.2f}   {recall:>6.2f}   {f1:>8.2f}   {support:>7}")

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    lines.append("-" * 68)
    lines.append(f"Accuracy: {accuracy*100:.2f}% ({total_correct}/{total_samples})")
    return "\n".join(lines), accuracy


def save_confusion_matrix_image(cm, labels_sorted, label_map, save_path):
    cell = 90
    left = 220
    top = 140
    n = len(labels_sorted)
    width = left + n * cell + 40
    height = top + n * cell + 80
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    cv2.putText(img, "Confusion Matrix", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    cv2.putText(img, "Hang = nhan dung | Cot = du doan", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)

    max_value = int(cm.max()) if cm.size > 0 else 1
    max_value = max(max_value, 1)

    for j, label_id in enumerate(labels_sorted):
        name = label_map[label_id][:10]
        x = left + j * cell + 5
        cv2.putText(img, name, (x, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    for i, label_id in enumerate(labels_sorted):
        name = label_map[label_id][:18]
        y = top + i * cell + cell // 2
        cv2.putText(img, name, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        for j in range(n):
            value = int(cm[i, j])
            intensity = int(255 - 170 * value / max_value)
            color = (255, intensity, intensity)
            x1 = left + j * cell
            y1 = top + i * cell
            x2 = x1 + cell
            y2 = y1 + cell
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
            cv2.putText(img, str(value), (x1 + 32, y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    cv2.imwrite(str(save_path), img)


def train_model():
    init_folders()
    check_cv2_face()

    images, labels, label_map = load_dataset()
    print("\nTHONG TIN DATASET")
    print("Tong so anh:", len(images))
    print("So nguoi:", len(label_map))
    print("Nhan:", label_map)

    # Chia train/test de co ket qua danh gia.
    train_idx, test_idx = make_train_test_split(labels, test_ratio=0.2, seed=42)

    if len(test_idx) == 0:
        raise RuntimeError("Khong du anh de chia tap test. Moi nguoi nen co toi thieu 20 anh, tot nhat 150-300 anh.")

    X_train = [images[i] for i in train_idx]
    y_train = labels[train_idx]
    X_test = [images[i] for i in test_idx]
    y_test = labels[test_idx]

    model = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )
    model.train(X_train, y_train)

    y_pred = []
    confidences = []
    for img in X_test:
        pred_label, confidence = model.predict(img)
        y_pred.append(pred_label)
        confidences.append(confidence)

    y_pred = np.array(y_pred, dtype=np.int32)
    labels_sorted = sorted(label_map.keys())
    cm = build_confusion_matrix(y_test, y_pred, labels_sorted)
    report, acc = make_report_text(y_test, y_pred, labels_sorted, label_map, cm)

    print("\nKET QUA DANH GIA")
    print("Do chinh xac test:", round(acc * 100, 2), "%")
    print("\nBao cao danh gia:\n", report)
    print("\nConfusion matrix:\n", cm)

    # Train lai tren toan bo dataset de demo camera tot hon.
    final_model = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )
    final_model.train(images, labels)

    model_path = MODEL_DIR / "face_lbph_model.yml"
    labels_path = MODEL_DIR / "label_map.json"
    final_model.write(str(model_path))

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, ensure_ascii=False, indent=4)

    report_path = OUTPUT_DIR / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PROJECT: Nhan dien khuon mat bang camera\n")
        f.write(f"Tong so anh: {len(images)}\n")
        f.write(f"So nguoi: {len(label_map)}\n")
        f.write(f"So anh train: {len(train_idx)}\n")
        f.write(f"So anh test: {len(test_idx)}\n")
        f.write(f"Do chinh xac test: {acc*100:.2f}%\n")
        if confidences:
            f.write(f"Confidence trung binh tren test: {np.mean(confidences):.2f}\n\n")
        f.write("Label map:\n")
        f.write(json.dumps({str(k): v for k, v in label_map.items()}, ensure_ascii=False, indent=4))
        f.write("\n\nBao cao danh gia:\n")
        f.write(report)
        f.write("\n\nConfusion matrix:\n")
        f.write(str(cm))

    fig_path = OUTPUT_DIR / "confusion_matrix.png"
    save_confusion_matrix_image(cm, labels_sorted, label_map, fig_path)

    print("\nDA LUU KET QUA")
    print("Model:", model_path)
    print("Label map:", labels_path)
    print("Bao cao danh gia:", report_path)
    print("Confusion matrix:", fig_path)


# =========================
# 6. LOAD MODEL
# =========================
def load_model_and_labels():
    check_cv2_face()

    model_path = MODEL_DIR / "face_lbph_model.yml"
    labels_path = MODEL_DIR / "label_map.json"

    if not model_path.exists() or not labels_path.exists():
        raise RuntimeError("Chua co model. Hay chay chuc nang train model truoc.")

    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(str(model_path))

    with open(labels_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    label_map = {int(k): v for k, v in label_map.items()}
    return model, label_map


# =========================
# 7. DEMO CAMERA REAL-TIME
# =========================
def demo_camera():
    model, label_map = load_model_and_labels()
    detector = get_face_detector()
    cap = open_camera()

    print("\nDANG DEMO CAMERA")
    print("- Khung xanh: nguoi da nhan dien")
    print("- Khung do: Unknown")
    print("- Bam Q de thoat")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Khong doc duoc khung hinh tu camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)

        for box in faces:
            x, y, w, h = box
            face_img = preprocess_face(gray, box)
            label_id, confidence = model.predict(face_img)

            if confidence <= UNKNOWN_THRESHOLD:
                name = label_map.get(label_id, "Unknown")
                text = name 
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Q: quit", (10, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Demo nhan dien khuon mat", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================
# 8. TEST 1 ANH CO SAN
# =========================
def predict_image():
    model, label_map = load_model_and_labels()
    detector = get_face_detector()

    image_path = input("Nhap duong dan anh can test: ").strip().strip('"')
    image_path = Path(image_path)

    if not image_path.exists():
        print("Khong tim thay anh:", image_path)
        return

    img = cv2.imread(str(image_path))
    if img is None:
        print("OpenCV khong doc duoc anh.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, detector)

    if not faces:
        print("Khong tim thay khuon mat trong anh.")
        print("Van hien thi anh da thu nho de ban kiem tra lai.")
        hien_thi_anh_vua_man_hinh("Ket qua test anh", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    for box in faces:
        x, y, w, h = box
        face_img = preprocess_face(gray, box)
        label_id, confidence = model.predict(face_img)

        if confidence <= UNKNOWN_THRESHOLD:
            name = label_map.get(label_id, "Unknown")
            text = f"{name} ({confidence:.1f})"
            color = (0, 255, 0)
        else:
            text = f"Unknown ({confidence:.1f})"
            color = (0, 0, 255)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Tranh chu bi vuot ra ngoai anh neu mat nam gan sat mep tren.
        text_y = max(y - 10, 30)
        cv2.putText(
            img,
            text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2
        )

    print("Da hien thi ket qua test anh.")
    print("Nhan phim bat ky tren cua so anh de dong.")

    # Diem da sua: khong hien thi anh goc qua lon nua, chi hien thi ban da resize.
    hien_thi_anh_vua_man_hinh("Ket qua test anh", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================
# 9. KIEM TRA DATASET
# =========================
def show_dataset_info():
    init_folders()
    print("\nTHONG TIN THU MUC PROJECT")
    print("BASE_DIR:", BASE_DIR)
    print("DATASET_DIR:", DATASET_DIR)
    print("MODEL_DIR:", MODEL_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    person_dirs = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])
    if not person_dirs:
        print("\nChua co du lieu. Hay chon muc 1 de thu thap anh khuon mat.")
        return

    print("\nDATASET HIEN CO:")
    total = 0
    for p in person_dirs:
        n = len(list(p.glob("*.jpg"))) + len(list(p.glob("*.png")))
        total += n
        print(f"- {p.name}: {n} anh")
    print("Tong so anh:", total)


# =========================
# 10. MENU CHAY TREN SPYDER
# =========================
def main():
    init_folders()

    menu = """
================ PROJECT NHAN DIEN KHUON MAT ================
1. Thu thap du lieu khuon mat bang camera
2. Huan luyen model + danh gia
3. Demo nhan dien bang camera
4. Test model voi 1 anh
5. Xem thong tin dataset
0. Thoat
=============================================================
"""

    while True:
        print(menu)
        choice = input("Nhap lua chon: ").strip()

        try:
            if choice == "1":
                collect_faces()
            elif choice == "2":
                train_model()
            elif choice == "3":
                demo_camera()
            elif choice == "4":
                predict_image()
            elif choice == "5":
                show_dataset_info()
            elif choice == "0":
                print("Da thoat chuong trinh.")
                break
            else:
                print("Lua chon khong hop le.")
        except Exception as e:
            print("\nDA XAY RA LOI:")
            print(e)
            print("\nGoi y:")
            print("- Neu thieu cv2.face: chay tai thu vien.py roi restart kernel Spyder")
            print("- Neu khong mo duoc camera: dong app dang dung webcam, doi CAMERA_IDS = [1, 0, 2]")
            print("- Neu chua train duoc: can toi thieu 2 nguoi, moi nguoi nen co 150-300 anh")


if __name__ == "__main__":
    main()
