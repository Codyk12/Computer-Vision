import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from time import time


def get_fore_back(im):
    """
    Function: get_fore_back
    -----------------------
    Opens a GUI that can be used to annotate the given image with foreground and background seeds
    
    Args:
    -----
        im: The input image can either be of type uint8 with range [0,255] or float with range[0, 1]
    
    Returns:
    --------
        a tuple containing locations for the foreground seeds and a tuple containing locations for the background seeds
        
    Example:
    --------
    >> fore, back = get_fore_back(img)
    >> img[fore]  # the foreground pixels
    >> img[back]  # the background pixels
    """
    if im.dtype != np.uint8 and np.max(im) <= 1.0 and np.min(im) >= 0:
        img = np.uint8(255 * im)
    elif im.dtype == np.uint8 and np.max(im) <= 255 and np.min(im) >= 0:
        img = im.copy()
    else:
        raise ValueError("Input image must be either dtype uint8 with range [0, 255] or float with range [0, 1] but got im.dtype = {}".format(im.dtype))
        
        
    ground_num = 1
    anno = np.zeros(img.shape[:-1])
    lastx = 0
    lasty = 0
    colors = {
        1:"red",
        2:"blue"
    }
    
    def reset():
        nonlocal anno, photo
        anno = np.zeros(img.shape[:2])
        canvas.create_image(img.shape[1]//2, img.shape[0]//2, image=photo)
        
    def set_ground_num(num):
        nonlocal ground_num
        ground_num = num
        
    def mouse_down(event):
        nonlocal anno, lastx, lasty, canvas, colors, ground_num
        h, w = anno.shape
        lastx = min(max(event.x, 0), w - 1)
        lasty = min(max(event.y, 0), h - 1)
        x = min(max(event.x - 1, 0), w - 1)
        y = min(max(event.y - 1, 0), h - 1)
        canvas.create_line(lastx, lasty, x, y, fill=colors[ground_num])
        rr, cc = draw.line(lastx, lasty, x , y)
        anno[cc, rr] = ground_num

    def mouse_drag(event):
        nonlocal anno, lastx, lasty, canvas, colors, ground_num
        h, w = anno.shape
        x = min(max(event.x, 0), w - 1)
        y = min(max(event.y, 0), h - 1)
        canvas.create_line(lastx, lasty, x, y, fill=colors[ground_num])
        rr, cc = draw.line(lastx, lasty, x , y)
        anno[cc, rr] = ground_num
        lastx = x
        lasty = y
        
    root = tk.Tk()
    
    btn_frame = tk.Frame(root)
    btn_frame.pack(side=tk.TOP)
    
    reset_btn = tk.Button(btn_frame, text='Reset',command=reset)
    reset_btn.pack(side=tk.LEFT)
    
    fore_btn = tk.Button(btn_frame, text='Foreground', command=lambda: set_ground_num(1))
    fore_btn.pack(side=tk.LEFT)
    
    back_btn = tk.Button(btn_frame, text='Background', command=lambda: set_ground_num(2))
    back_btn.pack(side=tk.LEFT)
    
    return_btn = tk.Button(btn_frame, text='Return', command=root.destroy)
    return_btn.pack(side=tk.LEFT)
    
    canvas = tk.Canvas(root, width=img.shape[1], height=img.shape[0])
    canvas.bind("<Button-1>", mouse_down)
    canvas.bind("<B1-Motion>", mouse_drag)
    photo = ImageTk.PhotoImage(Image.fromarray(img))
    canvas.pack(side=tk.BOTTOM)
    reset()
    root.mainloop()
    rtn = np.where(anno == 1), np.where(anno == 2)
    np.save('user_data/generated" + str(time()) + ".npy', rtn)
    return rtn
    
if __name__ == '__main__':
    print(get_fore_back(plt.imread('provided_images/simplecircle.png')))
