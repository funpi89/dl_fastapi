import uvicorn
from fastapi import FastAPI
from routers import image_caption, object_detect

app = FastAPI()


app.include_router(image_caption.router)
app.include_router(object_detect.router)

if __name__ == "__main__":
    uvicorn.run(app, debug=True)