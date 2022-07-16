from pathlib import Path
import os
import pdb
import cv2
import argparse
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument('--directory', '-d', type=Path, help='Path to photos')
args = ap.parse_args()

out_dir = args.directory / 'out'
out_dir.mkdir(exist_ok=True, parents=True)

def proc_frame(img: Image) -> Image:
    img = ImageOps.exif_transpose(img)
    size = max(img.width, img.height)
    final = img.resize((size, size)).filter(ImageFilter.GaussianBlur(radius=100))
    xstart = (size - img.width) // 2
    ystart = (size - img.height) // 2
    final = np.array(final)
    final[ystart:ystart+img.height, xstart:xstart+img.width] = np.asarray(img)
    return Image.fromarray(final)

def process_image(path: Path, output_path: Path):
    try:
        img = Image.open(path)
    except:
        print('- [ERROR] Failed to read', str(path))
        return
    final = proc_frame(img)
    final.save(output_path)
    

def process_video(path: Path, output_path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print('- [ERROR] Failed to read', str(path))
        return
    
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
    t = tqdm(total=length*2, desc=str(input_path), leave=False, dynamic_ncols=True)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # rotate frame because cv2 doesn't respect EXIF data
        if frame.shape[0] < frame.shape[1]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        final = proc_frame(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frames.append(np.asarray(final))
        t.update()
    cap.release()

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path.parent / f'{output_path.stem}.mp4'), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        t.update()
    writer.release()
    t.close()


if __name__ == '__main__':
    for f in os.listdir(args.directory):
        input_path = args.directory / f
        if not input_path.is_file():
            continue
        output_path = out_dir / f
        if input_path.suffix[1:].lower() not in ['jpg', 'jpeg', 'png']:
            process_video(input_path, output_path)
        else:
            process_image(input_path, output_path)
        print('- Finished', str(output_path))
    
    
    
    


