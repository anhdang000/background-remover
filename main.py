from fastapi import FastAPI
from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
import io
from starlette.responses import StreamingResponse
from predict import predict_img, mask_to_image
import cv2
import uvicorn

from unet import UNet
from utils.dataset import BasicDataset

from libs import *


app = FastAPI()

@app.post('/challenge')
def predict(challenge: str = Form(...), input: UploadFile = File(...)):
    # Check challenge name
    if challenge != 'cv3':
        return {"message": "The API only works for challenge `cv3`"}

    # Check file format
    file_ext = input.filename.split('.')[-1]
    supported_image_formats = ['bmp', 'jpg', 'jpeg', 'jp2', 'png', 'tiff', 'webp', 'xbm']
    if file_ext not in supported_image_formats:
        return {
            "error": "Cannot read the imported file. Please refer to our supported image formats", 
            "supported_image_formats": str(supported_image_formats)
            }

    # Load model U-Net
    net = UNet(n_channels=3, n_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Check file validation
    try:
        img = Image.open(input.file).convert('RGB')
    except:
        return {"error": f"File {input.filename} could not be read"}

    if img is None:
        return {"error": f"File {input.filename} could not be read"}
    
    # Compute scale factor that fix both dimentions
    w, h = img.size
    scale_factor = [INPUT_SIZE/max(w, h), INPUT_SIZE/max(w, h)]
    newW = w * scale_factor[0]
    newH = h * scale_factor[1]
    
    while newW < 50:
        scale_factor[0] *= 2
        newW = w * scale_factor[0]
    while newH < 50:
        scale_factor[1] *= 2
        newH = h * scale_factor[1]

    img_np = np.array(img)
    img_np = cv2.resize(img_np, dsize=None, fx=scale_factor[0], fy=scale_factor[1])
    
    mask = predict_img(
        net,
        img,
        device,
        scale_factor,
        out_threshold=0.5,
        )
    
    # Retrieve final result
    mask = np.resize(mask, img_np.shape[:-1])
    mask_rgb = np.stack([mask == 1]*3, axis=2)

    image_bg_removed = np.where(mask_rgb, img_np, 255)
    image_bg_removed = cv2.cvtColor(image_bg_removed, cv2.COLOR_BGR2RGB)

    _, im_png = cv2.imencode('.png', image_bg_removed)

    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')