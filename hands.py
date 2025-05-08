import os
os.environ['GLOG_minloglevel'] = '2'
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2
import mediapipe

from tkinter import filedialog

NUMBERS = [4, 8, 12, 16, 20]
VALLEYS = [2, 5, 9, 13, 17]

class ImageSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Выбор изображения")
        
        self.image_folder = None
        
        self.selected_image = None
        
        self.folder_button = tk.Button(root, text="Выбрать папку", command=self.choose_folder)
        self.folder_button.pack(pady=10)
        
        self.canvas = tk.Canvas(root)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.image_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_frame, anchor=tk.NW)
        
        self.image_frame.bind("<Configure>", self.on_frame_configure)
        
        self.images = []
        self.thumbnails = []

    def choose_folder(self):
        self.image_folder = filedialog.askdirectory()  
        if self.image_folder:
            self.load_images()

    def load_images(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        self.images.clear()
        self.thumbnails.clear()
        
        self.image_files = sorted(
            [f for f in os.listdir(self.image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))],
            key=lambda x: x.lower()  
        )
        
        if not self.image_files:
            return
        
        for idx, image_file in enumerate(self.image_files):
            image_path = os.path.join(self.image_folder, image_file)
            try:
                image = Image.open(image_path)
                image.thumbnail((100, 100)) 
                photo = ImageTk.PhotoImage(image)
                
                self.images.append(photo)
                
                frame = tk.Frame(self.image_frame)
                frame.grid(row=idx // 10, column=idx % 10, padx=5, pady=5)  
                
                button = tk.Button(frame, image=photo, command=lambda f=image_file: self.select_image(f))
                button.pack()
                
                label = tk.Label(frame, text=image_file)
                label.pack()
            except Exception as e:
                print(f"Ошибка загрузки изображения {image_file}: {e}")

    def select_image(self, image_file):
        self.selected_image = image_file
        self.root.destroy() 

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

def show_image(image, title):
    image_pil = Image.fromarray(image).convert("RGB")
    image_pil.thumbnail((500, 500))
    image_tk = ImageTk.PhotoImage(image_pil)
    mask_label = tk.Label(root, image=image_tk, text=title, compound="top")
    mask_label.image = image_tk
    mask_label.pack(side="left", padx=10, pady=10)

def cosine(a, b):
  return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)

def normal(start, end, far):
  cos = cosine(far - start, end - start)
  return np.sqrt(1 - cos ** 2) * np.linalg.norm(far - start).item()

def process_hand(image):
  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
  image = clahe.apply(image[:,:,0])
  _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  mask = ~mask
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
  mask = cv2.dilate(mask, kernel, iterations=10)
  show_image(mask, "Маска")
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  max_contour = max(contours, key=cv2.contourArea)
  hull = cv2.convexHull(max_contour, returnPoints=False)
  defects = cv2.convexityDefects(max_contour, hull)
  triples = []
  for i, defect in enumerate(defects):
    s, e, f, d = defect[0]
    start = max_contour[s][0]
    end = max_contour[e][0]
    far = max_contour[f][0]
    dist = np.sqrt((start - far) ** 2 + (end - far) ** 2).sum()
    if dist < 30:
      continue
    n = normal(start, end, far)
    if n < 100:
      continue
    triples.append([start, end, far])
  return triples

def process_image(path, label):
    hands = mediapipe.solutions.hands.Hands()
    name = path
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    show_image(image, "Исходное изображение")
    image1 = image.copy()
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mediapipe.solutions.drawing_utils.draw_landmarks(
                image, hand_landmarks, mediapipe.solutions.hands.HAND_CONNECTIONS)
    landmarks = []
    for landmark in results.multi_hand_landmarks[0].landmark:
        landmarks.append(np.array([int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])]))
    for i in NUMBERS:
      cv2.circle(image, landmarks[i], 5, (255, 0, 0), 3)
    for i in VALLEYS:
      cv2.circle(image, landmarks[i], 5, (0, 255, 0), 3)
    triples = process_hand(image1.copy())
    new_triples = []
    for j, t in enumerate(triples):
      dists = []
      for i in range(len(NUMBERS) - 1):
        dists.append(np.linalg.norm(t[0] - landmarks[NUMBERS[i]]) + np.linalg.norm(t[1] - landmarks[NUMBERS[i + 1]]))
        dists.append(np.linalg.norm(t[1] - landmarks[NUMBERS[i]]) + np.linalg.norm(t[0] - landmarks[NUMBERS[i + 1]]))
      min_idx = np.array(dists).argmin()
      if dists[min_idx] > 200:
        triples[j] = None
        continue
      triples[j][min_idx % 2] = landmarks[NUMBERS[min_idx // 2]]
      triples[j][(min_idx + 1) % 2] = landmarks[NUMBERS[(min_idx // 2) + 1]]
      new_triples.append((min_idx // 2).item())
    for t in triples:
      if t is not None:
        color = [150, 150, 150]
        cv2.line(image, t[0], t[2], thickness=10,color=color)
        cv2.line(image, t[2], t[1], thickness=10,color=color)
    show_image(image, "Скелет и зазоры")
    row = "1"
    for i in range(len(NUMBERS) - 1):
      if i in new_triples:
        row += "-"
      else:
        row += "+"
      row += f"{i+2}"
    points = f"!,{name.split('/')[-1]}"
    for n in NUMBERS:
      points += f",T {landmarks[n][0]} {landmarks[n][1]}"
    valleys = []
    for n in range(len(VALLEYS) - 1):
      valleys.append((landmarks[VALLEYS[n]] + landmarks[VALLEYS[n+1]]) // 2)
      points += f",V {valleys[-1][0]} {valleys[-1][1]}"
    points += ",?"
    for i in range(len(NUMBERS) - 1):
      cv2.line(image1, landmarks[NUMBERS[i]], valleys[i], thickness=3,color=(0,255,0))
      cv2.line(image1, valleys[i], landmarks[NUMBERS[i + 1]], thickness=3,color=(0,255,0))
    show_image(image1, "Пальцы")
    label.config(text=f"{row}")
    with open("./Results.txt", "w") as f:
      print(row, file=f)
      print(points, file=f)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Детектор пальцев")
    app = ImageSelectorApp(root)
    root.mainloop()
    
    if app.selected_image:
        path = f"{app.image_folder}/{app.selected_image}"
        root = tk.Tk()
        root.title("Детектор пальцев")

        label = tk.Label(root, text="", font=("Arial", 16))
        label.pack()

        process_image(path, label)

        root.mainloop()
    else:
        print("Изображение не выбрано.")
