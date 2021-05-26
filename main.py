from fastapi import FastAPI
from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
import io
from starlette.responses import StreamingResponse
from predict import predict_img, mask_to_image
import cv2
import uvicorn

from libs import *


app = FastAPI()

@app.get('/{challenge}')
async def read_challenge_option(challenge):
    return challenge

@app.post('/challenge')
def predict(challenge: str = Form(...), input: UploadFile = Form(...)):
    # Check validation
    file_ext = input.filename.split('.')[-1]
    supported_image_formats = ['bmp', 'jpg', 'jpeg', 'jp2', 'png', 'tiff', 'webp', 'xbm']
    if file_ext not in supported_image_formats:
        return {"error": "Cannot read the imported file. Please refer to our supported image formats", "supported_image_formats": str(supported_image_formats)}

    # Load model U-Net
    net = UNet(n_channels=3, n_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Retrieve final result
    try:
        img = Image.open(input.file).convert('RGB')
    except:
        return {"error": f"File {input.filename} could not be read"}

    img_np = np.array(img)
    mask = predict_img(net=net,
                        full_img=img,
                        scale_factor=0.5,
                        out_threshold=0.5,
                        device=device)
    mask = np.resize(mask, img_np.shape[:-1])
    mask_rgb = np.stack([mask == 1]*3, axis=2)

    image_bg_removed = np.where(mask_rgb, img_np, 255)
    image_bg_removed = cv2.cvtColor(image_bg_removed, cv2.COLOR_BGR2RGB)

    _, im_png = cv2.imencode('.png', image_bg_removed)

    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


@app.get("/")
async def main():
    content = """
<body>
<img src="output.jpg"/>
<form action="/api/predict" enctype="multipart/form-data" method="post">
<input name="files" type="file">
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')