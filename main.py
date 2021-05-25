from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
from predict import predict_img, mask_to_image
import uvicorn

from libs import *


app = FastAPI()

@app.get('/index')
def hello_world(name : str):
    return f"Hello {name}"

@app.post('/api/predict')
def predict(file: UploadFile = File(...)):
    file_ext = file.filename.split('.')[-1]
    net = UNet(n_channels=3, n_classes=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net.to(device=device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    img = Image.open(file.filename).convert('RGB')
    img_np = np.array(img)
    mask = predict_img(net=net,
                        full_img=img,
                        scale_factor=0.5,
                        out_threshold=0.5,
                        device=device)
    mask = np.resize(mask, img_np.shape[:-1])
    mask_rgb = np.stack([mask == 1]*3, axis=2)
    print(mask_rgb.shape)
    image_bg_removed = np.where(mask_rgb, img_np, 0)
    print(np.max(image_bg_removed))
    result = Image.fromarray(image_bg_removed)
    
    output_path = 'output.' + file_ext
    result.save(output_path)

    # if args.viz:
    #     logging.info("Visualizing results for image {}, close to continue ...".format(fn))
    #     plot_img_and_mask(img, mask)

    return FileResponse(output_path)

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')